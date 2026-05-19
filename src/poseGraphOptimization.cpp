#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <queue>
#include <fstream>
#include <csignal>
#include <optional>
#include <unistd.h>
#include <condition_variable>
#include <unordered_set>
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/utils.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_lib.h>
#include "solid/solid_module.h"
#include "fast_lio/msg/frame.hpp"
#include <std_srvs/srv/trigger.hpp>
#include "pose_graph_optimization/srv/save_map.hpp"
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/nano_gicp.hpp>

#include "patchwork/patchworkpp.h"
#include "CurvedVoxelClustering.hpp"

using namespace std;

using namespace gtsam;

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using CVCHandler = perception::CurvedVoxelClustering<pcl::PointXYZI>;

struct Pose6 {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

string save_directory, optimized_poseDirectory, odom_poseDirectory, edge_directory, ScansDirectory, DebugDirectory;
int curr_kf_idx = 0;
vector<nav_msgs::msg::Odometry> kf_poses;
SOLiDModule solidModule;
std::mutex mLoop, mLC, mViz, mBuf, mKF;
condition_variable sig_buffer;


// solid params
double R_SOLiD_THRES;
double FOV_u, FOV_d, VOXEL_SIZE;
int NUM_ANGLE, NUM_RANGE, NUM_HEIGHT;
int MIN_DISTANCE, MAX_DISTANCE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE;
queue<tuple<int, int, Eigen::Matrix4f>> solidLoopBuf; 
std::tuple<int, int, Eigen::Matrix4f> prev_loop_pair;

int num_scan;
double det_range_;
double blind;
double loop_dist;

// edge measurement params
pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter_map;
nano_gicp::NanoGICP<PointType2, PointType2> gicp;
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree (new pcl::KdTreeFLANN<pcl::PointXYZI>());
std::vector<int> indiceLet;
double dop_thres = 0;

//for pose graph
gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
bool isLoopClosed = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
gtsam::Vector odomNoiseVector6(6);
gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
double loopNoiseScore = 0.5;
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
int recentIdxUpdated = 0;

// Range image parameters
int horizontal_resolution = 1024;
double LIDAR_HOR_MIN = -180.0F;
double LIDAR_HOR_MAX = 180.0F;
std::unique_ptr<patchwork::PatchWorkpp> Patchworkpp_;
std::unique_ptr<patchwork::PatchWorkpp> Patchworkpp_Fine;

visualization_msgs::msg::Marker loopLine;
nav_msgs::msg::Path PGO_path;

fstream odom_stream, optimized_stream, edge_stream;
pcl::PointCloud<pcl::PointXYZI>::Ptr kf_nodes (new pcl::PointCloud<pcl::PointXYZI>());
std::vector<Pose6> keyframePoses;
std::vector<Pose6> keyframePosesUpdated;
std::vector<gtsam::Vector> keyframeCovariance;
std::vector<double> keyframeTimes;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kf_node_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr LoopLineMarker_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PubPGO_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr PubPGO_map;
rclcpp::Service<pose_graph_optimization::srv::SaveMap>::SharedPtr save_service_;
rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr cancel_service_;

// 저장 취소 플래그 (atomic: 다른 스레드에서 안전하게 설정 가능)
std::atomic<bool> g_cancel_save{false};

// save_trajectory_callback 내부에서 throw해 즉시 빠져나오기 위한 예외 타입
struct SaveCancelledException {};

void cancel_save_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    RCLCPP_INFO(rclcpp::get_logger("posegraphoptimization"), "Save map cancel requested");
    g_cancel_save = true;
    response->success = true;
    response->message = "Cancel signal sent to save_trajectory";
}

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1.0), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
} // initNoises
                
Eigen::MatrixXf PointCloud2ToEigenMat(const pcl::PointCloud<pcl::PointXYZI>::Ptr pc) 
{
    Eigen::MatrixXf points;
    size_t num_points = pc->points.size();
    points.resize(num_points, 3);

    for (size_t i = 0; i < num_points; i++) 
    {
        points.row(i) << pc->points[i].x, pc->points[i].y, pc->points[i].z;
    }

    return points;
}

gtsam::Pose3 Pose6toGTSAMPose3(const Pose6& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6toGTSAMPose3

Pose6 getOdom(nav_msgs::msg::Odometry _odom)
{
    auto tx = _odom.pose.pose.position.x;
    auto ty = _odom.pose.pose.position.y;
    auto tz = _odom.pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::msg::Quaternion quat = _odom.pose.pose.orientation;
    tf2::Matrix3x3(tf2::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6{tx, ty, tz, roll, pitch, yaw};
} // getOdom

Eigen::Matrix4f get_TF_Matrix(const Pose6 Pose)
{
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(Pose.yaw, Eigen::Vector3f::UnitZ())
             * Eigen::AngleAxisf(Pose.pitch, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(Pose.roll, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f TF(Eigen::Matrix4f::Identity());
    TF.block(0,0,3,3) = rotation;
    TF(0,3) = Pose.x;
    TF(1,3) = Pose.y;
    TF(2,3) = Pose.z;

    return TF;
}

double computeDOP(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, Eigen::Vector3d pos)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr dop_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterDOP;
    downSizeFilterDOP.setLeafSize(2, 2, 2);
    downSizeFilterDOP.setInputCloud(cloud);
    downSizeFilterDOP.filter(*dop_cloud);  
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indiceLet);
    indiceLet.clear();

    std::vector<Eigen::Vector3d> range_info;
    for (size_t k = 0; k < dop_cloud->points.size(); k++)
    {
        double r = sqrt(pow((dop_cloud->points[k].x-pos(0)), 2) + pow((dop_cloud->points[k].y-pos(1)), 2) + pow((dop_cloud->points[k].z-pos(2)), 2));
        if (r < blind)    continue;
        Eigen::Vector3d r_info;
        r_info(0) = dop_cloud->points[k].x / r;
        r_info(1) = dop_cloud->points[k].y / r;
        r_info(2) = dop_cloud->points[k].z / r;
        range_info.push_back(r_info);    
    }
    Eigen::MatrixXd AA(range_info.size(), 3);
    for (size_t p = 0; p < range_info.size(); p++)
    {
        AA(p, 0) = range_info[p](0);
        AA(p, 1) = range_info[p](1);
        AA(p, 2) = range_info[p](2);
    }
    Eigen::Matrix3d A_sq;
    Eigen::Matrix3d Q;
    A_sq = AA.transpose() * AA;
    Q = A_sq.inverse();

    double pdop = sqrt(Q(0, 0) + Q(1, 1) + Q(2, 2));
    if (pdop == 0 || pdop > 100 || std::isnan(pdop) == true)
    {
        pdop = 100;
    }
    return pdop;
}

std::optional<gtsam::Pose3> doGICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4f delta_TF)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr targetKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_curr_kf_idx) + ".pcd", *cureKeyframeCloud);
    pcl::io::loadPCDFile(ScansDirectory + std::to_string(_loop_kf_idx) + ".pcd", *targetKeyframeCloud);
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.setInputCloud(cureKeyframeCloud);
    downSizeFilter.filter(*cureKeyframeCloud);
    downSizeFilter.setInputCloud(targetKeyframeCloud);
    downSizeFilter.filter(*targetKeyframeCloud);

    gicp.setInputTarget(targetKeyframeCloud);
    gicp.setInputSource(cureKeyframeCloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    gicp.align(*aligned_cloud, delta_TF);
    Eigen::Matrix4f edge_TF = gicp.getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZI>::Ptr matchKeyframeCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cureKeyframeCloud, *matchKeyframeCloud, edge_TF);

    Eigen::Matrix<double, 6, 6> hessian = gicp.getHessian();
    
    typedef Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> EigenSolver;
    EigenSolver es;
    Eigen::Matrix<double, 6, 6> hessian_inv = hessian.inverse();
    es.compute(hessian_inv);
    
    Eigen::VectorXcd eigenvalues = es.eigenvalues();

    std::complex<double> max_eigenvalue = eigenvalues(0);
    for (int i = 1; i < eigenvalues.size(); ++i) 
    {
        if (eigenvalues(i).real() > max_eigenvalue.real()) {

            max_eigenvalue = eigenvalues(i);
        }
    }
    double max_eigen = sqrt(fabs(max_eigenvalue.real()));

    kdtree->setInputCloud(targetKeyframeCloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr MatchedCloud (new pcl::PointCloud<pcl::PointXYZI>());
    for (int k = 0; k < matchKeyframeCloud->points.size(); k++)
    {
        kdtree->nearestKSearch(matchKeyframeCloud->points[k], 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < 0.1)
        {
            MatchedCloud->points.push_back(matchKeyframeCloud->points[k]);
        }
    }

    double matching_dop = computeDOP(MatchedCloud, Eigen::Vector3d(edge_TF(0,3),edge_TF(1,3),edge_TF(2,3)));
    double curr_dop = computeDOP(cureKeyframeCloud, Eigen::Vector3d(0,0,0));
    double target_dop = computeDOP(targetKeyframeCloud, Eigen::Vector3d(0,0,0));    
    double max_dop;
    if (curr_dop > target_dop) 
    {
        max_dop = curr_dop;
    } 
    else 
    {
        max_dop = target_dop;
    }
    double dop_ratio = matching_dop / max_dop;

    pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, delta_TF);
    std::for_each(cureKeyframeCloud->points.begin(), cureKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 1.0; });
    std::for_each(targetKeyframeCloud->points.begin(), targetKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 2.0; });
    std::for_each(matchKeyframeCloud->points.begin(), matchKeyframeCloud->points.end(),
                  [](pcl::PointXYZI& point) { point.intensity = 3.0; });

    pcl::PointCloud<pcl::PointXYZI>::Ptr resultKeyframeCloud (new pcl::PointCloud<pcl::PointXYZI>());
    *resultKeyframeCloud += *cureKeyframeCloud;
    *resultKeyframeCloud += *targetKeyframeCloud;
    *resultKeyframeCloud += *matchKeyframeCloud;
    pcl::io::savePCDFileBinary(DebugDirectory + to_string(_curr_kf_idx) + "_" + to_string(dop_ratio) + "_" 
    + to_string(matching_dop) + ".pcd", *resultKeyframeCloud); // debug data
    if (dop_ratio < dop_thres && matching_dop < 1.0)
    {
        Eigen::Matrix3f edge_rot = edge_TF.block(0, 0, 3, 3);
        Eigen::Quaternionf final_q(edge_rot);
        // Get pose transformation
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf2::Quaternion(final_q.x(), final_q.y(), final_q.z(), final_q.w())).getRPY(roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(edge_TF(0,3), edge_TF(1,3), edge_TF(2,3)));

        loopNoiseScore = max_eigen; // constant is ok...
        robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(2.0), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

        isLoopClosed = true;
        return poseFrom.between(poseTo);
    }
    else
    {
        return std::nullopt;
    }
}

void kf_callback(const fast_lio::msg::Frame::SharedPtr kf_msg) 
{
    mBuf.lock();
    fast_lio::msg::Frame curr_kf = *kf_msg;
    curr_kf_idx = curr_kf.frame_idx;
    nav_msgs::msg::Odometry curr_pose = curr_kf.pose;
    kf_poses.push_back(curr_pose);

    Pose6 pose_curr = getOdom(curr_pose);
    keyframePoses.push_back(pose_curr);
    keyframePosesUpdated.push_back(pose_curr);

    sensor_msgs::msg::PointCloud2 pc_msg = curr_kf.pointcloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_kf_pc (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr curr_kf_pc_down (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(pc_msg, *curr_kf_pc);
    solidModule.down_sampling(*curr_kf_pc, curr_kf_pc_down);
    solidModule.makeAndSaveSolid(*curr_kf_pc_down);
         
    pcl::io::savePCDFileBinary(ScansDirectory + to_string(curr_kf_idx) + ".pcd", *curr_kf_pc); // scan data
    
    const int prev_node_idx = keyframePoses.size() - 2;
    const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
    
    if(!gtSAMgraphMade)
    {
        const int init_node_idx = 0;
        gtsam::Pose3 poseOrigin = Pose6toGTSAMPose3(keyframePoses.at(init_node_idx));

        // prior factor
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
        initialEstimate.insert(init_node_idx, poseOrigin);

        gtSAMgraphMade = true;
    }
    else
    {
        gtsam::Pose3 poseFrom = Pose6toGTSAMPose3(keyframePoses.at(prev_node_idx));
        gtsam::Pose3 poseTo = Pose6toGTSAMPose3(keyframePoses.at(curr_node_idx));
        // odom factor
        gtsam::Pose3 relPose = poseFrom.between(poseTo);

        odomNoiseVector6 << kf_poses[curr_node_idx].pose.covariance[0], 
                            kf_poses[curr_node_idx].pose.covariance[7], 
                            kf_poses[curr_node_idx].pose.covariance[14], 
                            kf_poses[curr_node_idx].pose.covariance[21], 
                            kf_poses[curr_node_idx].pose.covariance[28], 
                            kf_poses[curr_node_idx].pose.covariance[35];
        odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);
        keyframeCovariance.push_back(odomNoiseVector6);

        Eigen::Matrix4f to_TF = get_TF_Matrix(keyframePoses[curr_node_idx]);
        Eigen::Matrix4f from_TF = get_TF_Matrix(keyframePoses[prev_node_idx]);
        Eigen::Matrix4f delta_TF = from_TF.inverse() * to_TF;
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relPose, odomNoise));
        initialEstimate.insert(curr_node_idx, poseTo);                
    } 

    odom_stream << rclcpp::Time(curr_kf.header.stamp).seconds() << " "
                << curr_pose.pose.pose.position.x << " " << curr_pose.pose.pose.position.y << " " << curr_pose.pose.pose.position.z << " " 
                << curr_pose.pose.pose.orientation.x << " " << curr_pose.pose.pose.orientation.y << " " << curr_pose.pose.pose.orientation.z << " " << curr_pose.pose.pose.orientation.w <<endl;
    
    pcl::PointXYZI kf_node;
    kf_node.x = curr_pose.pose.pose.position.x;
    kf_node.y = curr_pose.pose.pose.position.y;
    kf_node.z = curr_pose.pose.pose.position.z;
    kf_node.intensity = curr_kf_idx;
    kf_nodes->points.push_back(kf_node);

    sensor_msgs::msg::PointCloud2 kf_nodes_msg;
    pcl::toROSMsg(*kf_nodes, kf_nodes_msg);
    kf_nodes_msg.header.stamp = curr_pose.header.stamp;
    kf_nodes_msg.header.frame_id = "odom";
    kf_node_pub->publish(kf_nodes_msg);  

    double timeLaserOdometry = rclcpp::Time(curr_kf.pose.header.stamp).seconds();
    keyframeTimes.push_back(timeLaserOdometry);
    mBuf.unlock();
}

void updatePoses(void)
{
    PGO_path.poses.clear();
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
        keyframePosesUpdated[node_idx] = p;

        geometry_msgs::msg::PoseStamped poseStampPGO;
        poseStampPGO.header.frame_id = "odom";
        poseStampPGO.pose.position.x = p.x;
        poseStampPGO.pose.position.y = p.y;
        poseStampPGO.pose.position.z = p.z;
        tf2::Quaternion quat_tf2;
        quat_tf2.setRPY(p.roll, p.pitch, p.yaw);
        poseStampPGO.pose.orientation = tf2::toMsg(quat_tf2);
        PGO_path.header.frame_id = "odom";
        PGO_path.poses.push_back(poseStampPGO);
        PGO_path.poses[node_idx].header.stamp = poseStampPGO.header.stamp;
    }
    PubPGO_path->publish(PGO_path);
}

void runISAM2opt(void)
{
    // called when a variable added
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    if (isLoopClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isLoopClosed = false;
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    recentIdxUpdated = int(isamCurrentEstimate.size());
    updatePoses();
}

void performSOLiDLoopClosure(void)
{
    if( int(keyframePoses.size()) < solidModule.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = solidModule.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SOLiDclosestHistoryFrameID = std::get<1>(detectResult);
    if( SOLiDclosestHistoryFrameID != -1 ) 
    { 
        const int prev_node_idx = SOLiDclosestHistoryFrameID;
        const int curr_node_idx = std::get<0>(detectResult); // because cpp starts 0 and ends n-1
        Eigen::Vector3d dist_vec;
        dist_vec(0) = keyframePoses[curr_node_idx].x - keyframePoses[prev_node_idx].x;
        dist_vec(1) = keyframePoses[curr_node_idx].y - keyframePoses[prev_node_idx].y;
        dist_vec(2) = keyframePoses[curr_node_idx].z - keyframePoses[prev_node_idx].z;
        double dist = dist_vec.norm();

        if (dist > loop_dist) return;

        // Eigen::Matrix4f to_TF = get_TF_Matrix(keyframePoses[curr_node_idx]);
        // Eigen::Matrix4f from_TF = get_TF_Matrix(keyframePoses[prev_node_idx]);
        // Eigen::Matrix4f delta_TF = from_TF.inverse() * to_TF;
        Eigen::Matrix4f delta_TF (Eigen::Matrix4f::Identity());
        Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(std::get<2>(detectResult), Eigen::Vector3f::UnitZ())
                     * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
                     * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());
        delta_TF.block(0,0,3,3) = rotation;

        if (solidLoopBuf.size() <= 10)
        {
            solidLoopBuf.push(std::make_tuple(prev_node_idx, curr_node_idx, delta_TF));
        }        
    }
} // performSOLiDLoopClosure


void process_lcd(void)
{
    rclcpp::Rate rate(2.0); 
    while (rclcpp::ok())
    {
        mLC.lock();
        performSOLiDLoopClosure();
        mLC.unlock();
        rate.sleep();
    }
} // process_lcd

void process_edge(void)
{
    while (rclcpp::ok())
    {
        while(!solidLoopBuf.empty())
        {
            mLoop.lock();
            std::tuple<int, int, Eigen::Matrix4f> loop_idx_pair = solidLoopBuf.front();
            solidLoopBuf.pop();
            const int prev_node_idx = get<0>(loop_idx_pair);
            const int curr_node_idx = get<1>(loop_idx_pair);
            const Eigen::Matrix4f delta_TF = get<2>(loop_idx_pair);
            auto relative_pose_optional = doGICPVirtualRelative(prev_node_idx, curr_node_idx, delta_TF);
            
            if(relative_pose_optional)
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                geometry_msgs::msg::Point p;
                p.x = keyframePoses[prev_node_idx].x;    p.y = keyframePoses[prev_node_idx].y;    p.z = keyframePoses[prev_node_idx].z;
                loopLine.points.push_back(p);
                p.x = keyframePoses[curr_node_idx].x;    p.y = keyframePoses[curr_node_idx].y;    p.z = keyframePoses[curr_node_idx].z;
                loopLine.points.push_back(p);
                LoopLineMarker_pub->publish(loopLine);
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));

                if(get<0>(prev_loop_pair) != prev_node_idx && get<1>(prev_loop_pair) != curr_node_idx)
                {
                    edge_stream << prev_node_idx << " " << curr_node_idx << " " << relative_pose.translation().x() << " " << relative_pose.translation().y() << " " 
                    << relative_pose.translation().z() << " " << relative_pose.rotation().roll() << " " << relative_pose.rotation().pitch() << " " 
                    << relative_pose.rotation().yaw() << " " << loopNoiseScore << " " << loopNoiseScore << " " << loopNoiseScore << " " << loopNoiseScore << " " 
                    << loopNoiseScore << " " << loopNoiseScore << endl;
                }
            }
            prev_loop_pair = loop_idx_pair;
            mLoop.unlock();
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

void process_optimization(void)
{
    float hz = 0.5;
    rclcpp::Rate rate(hz); // 0.5Hz = 2초 간격
    while (rclcpp::ok())
    {
        rate.sleep();
        if(gtSAMgraphMade)
        {
            mBuf.lock();
            runISAM2opt();
            mBuf.unlock();            
        }
    }
}

void process_viz(void)
{
    while(rclcpp::ok())
    {
        mViz.lock();
        pcl::PointCloud<pcl::PointXYZI>::Ptr VizMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
        VizMapCloud->points.clear();
        for (int i = 0; i < recentIdxUpdated; i++)
        {   
            double dist = sqrt(pow(keyframePosesUpdated.back().x-keyframePosesUpdated[i].x,2)
                            +pow(keyframePosesUpdated.back().y-keyframePosesUpdated[i].y,2)
                            +pow(keyframePosesUpdated.back().z-keyframePosesUpdated[i].z,2));
            if (dist > 50)  continue;
            Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(keyframePosesUpdated[i].yaw, Eigen::Vector3f::UnitZ())
                     * Eigen::AngleAxisf(keyframePosesUpdated[i].pitch, Eigen::Vector3f::UnitY())
                     * Eigen::AngleAxisf(keyframePosesUpdated[i].roll, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q(rotation);        
            Eigen::Matrix4f TF (Eigen::Matrix4f::Identity());
            TF.block(0,0,3,3) = rotation;
            TF(0,3) = keyframePosesUpdated[i].x;
            TF(1,3) = keyframePosesUpdated[i].y;
            TF(2,3) = keyframePosesUpdated[i].z;
            
            pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + ".pcd", *cureKeyframeCloud);
            
            pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, TF);
            *VizMapCloud += *cureKeyframeCloud; 
            downSizeFilter_map.setLeafSize(0.4, 0.4, 0.4);
            downSizeFilter_map.setInputCloud(VizMapCloud);
            downSizeFilter_map.filter(*VizMapCloud);
        }
        sensor_msgs::msg::PointCloud2 map_msg;
        pcl::toROSMsg(*VizMapCloud, map_msg);
        map_msg.header.frame_id = "odom";
        PubPGO_map->publish(map_msg);
        mViz.unlock();

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
            
}

Eigen::Matrix4f createTransformMatrix(const Pose6& pose)
{
    Eigen::Matrix3f rotation = (Eigen::AngleAxisf(pose.yaw, Eigen::Vector3f::UnitZ())
                              * Eigen::AngleAxisf(pose.pitch, Eigen::Vector3f::UnitY())
                              * Eigen::AngleAxisf(pose.roll, Eigen::Vector3f::UnitX())).toRotationMatrix();
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0,0,3,3) = rotation;
    transform(0,3) = pose.x;
    transform(1,3) = pose.y;
    transform(2,3) = pose.z;
    
    return transform;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr loadPointCloud(const std::string& filepath)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile(filepath, *cloud) == -1) {
        RCLCPP_ERROR(rclcpp::get_logger("posegraphoptimization"), "Failed to load point cloud: %s", filepath.c_str());
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr generateOptimizedMap()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr optimizedMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> downSizeMapFilter;
    downSizeMapFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);

    kf_nodes->points.clear();
    for (int k = 0; k < keyframePosesUpdated.size(); k++)
    {
        if (g_cancel_save.load()) throw SaveCancelledException{};

        Eigen::Matrix4f TF = createTransformMatrix(keyframePosesUpdated[k]);

        pcl::PointXYZI kf_node;
        kf_node.x = keyframePosesUpdated[k].x;
        kf_node.y = keyframePosesUpdated[k].y;
        kf_node.z = keyframePosesUpdated[k].z;
        kf_node.intensity = k;
        kf_nodes->points.push_back(kf_node);


        pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud = loadPointCloud(ScansDirectory + std::to_string(k) + ".pcd");
        if (cureKeyframeCloud->empty()) continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*cureKeyframeCloud, *transformedCloud, TF);

        // NaN 제거
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*transformedCloud, *transformedCloud, indices);

        // 개별 클라우드 다운샘플링
        pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZI>());
        downSizeMapFilter.setInputCloud(transformedCloud);
        downSizeMapFilter.filter(*downsampledCloud);

        *optimizedMapCloud += *downsampledCloud;
        printf("Processed %d, total points: %lu\n", k, optimizedMapCloud->size());
        fflush(stdout);
    }
    sensor_msgs::msg::PointCloud2 kf_nodes_msg;
    pcl::toROSMsg(*kf_nodes, kf_nodes_msg);
    kf_nodes_msg.header.stamp = get_ros_time(keyframeTimes.back());
    kf_nodes_msg.header.frame_id = "odom";
    kf_node_pub->publish(kf_nodes_msg);  

    // 최종 다운샘플링 (메모리 체크 추가)
    printf("Final downsampling: %lu points\n", optimizedMapCloud->size());
    fflush(stdout);    

    pcl::PointCloud<pcl::PointXYZI>::Ptr finalMapCloud(new pcl::PointCloud<pcl::PointXYZI>());

    // NaN 체크 및 제거
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*optimizedMapCloud, *optimizedMapCloud, indices);

    downSizeMapFilter.setInputCloud(optimizedMapCloud);
    downSizeMapFilter.filter(*finalMapCloud);

    printf("Final map size: %lu points\n", finalMapCloud->size());
    fflush(stdout);

    return finalMapCloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr generateStaticMap()
{    
    // Clear screen and show initial message
    std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
    std::cout << "Generating static map with optimized processing..." << std::endl;
    
    int total_frames = keyframePosesUpdated.size();    
    
    const int batch_size = 1; // Process 4 frames at a time to reduce memory usage
    int num_batches = (total_frames + batch_size - 1) / batch_size;
    
    for (int k = 0; k < total_frames; k++) 
    {
        if (g_cancel_save.load()) throw SaveCancelledException{};

        // Create thread-local copies to avoid sharing issues
        Eigen::Matrix4f TF = createTransformMatrix(keyframePosesUpdated[k]);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud = loadPointCloud(ScansDirectory + std::to_string(k) + ".pcd");

        if (cureKeyframeCloud->empty())
        {
            printf("Warning: Frame %d is empty\n", k);
            continue;
        }
        
        // NaN 제거
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, indices);

        // NaN 제거 후 포인트 클라우드가 비어있는지 확인
        if (cureKeyframeCloud->empty())
        {
            printf("Warning: Frame %d is empty after NaN removal\n", k);
            continue;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cureGroundCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureNonGroundCloud(new pcl::PointCloud<pcl::PointXYZI>());

        auto cloud = PointCloud2ToEigenMat(cureKeyframeCloud);
        Patchworkpp_->estimateGround(cloud);

        auto ground_indices = Patchworkpp_->getGroundIndices();
        auto nonground_indices = Patchworkpp_->getNongroundIndices();
        double time_taken = Patchworkpp_->getTimeTaken();

        pcl::PointCloud<pcl::PointXYZI>::Ptr tempGroundCloud(new pcl::PointCloud<pcl::PointXYZI>());
        // Ground와 non-ground 포인트 분리
        for (int idx = 0; idx < ground_indices.size(); idx++) 
        {
            int point_idx = ground_indices[idx];
            if (point_idx >= 0 && point_idx < cureKeyframeCloud->size()) 
            {
                tempGroundCloud->push_back(cureKeyframeCloud->points[point_idx]);
            }
        }

        for (int idx = 0; idx < nonground_indices.size(); idx++)
        {
            int point_idx = nonground_indices[idx];
            if (point_idx >= 0 && point_idx < cureKeyframeCloud->size()) 
            {
                cureNonGroundCloud->push_back(cureKeyframeCloud->points[point_idx]);
            }
        }

        // tempGroundCloud가 비어있는지 확인
        if (tempGroundCloud->empty())
        {
            printf("Warning: Frame %d has no ground points\n", k);
            continue;
        }

        cloud = PointCloud2ToEigenMat(tempGroundCloud);
        Patchworkpp_Fine->estimateGround(cloud);
        ground_indices = Patchworkpp_Fine->getGroundIndices();
        nonground_indices = Patchworkpp_Fine->getNongroundIndices();
        // Ground와 non-ground 포인트 분리
        for (int idx = 0; idx < ground_indices.size(); idx++) 
        {
            int point_idx = ground_indices[idx];
            if (point_idx >= 0 && point_idx < tempGroundCloud->size()) 
            {
                cureGroundCloud->push_back(tempGroundCloud->points[point_idx]);
            }
        }

        for (int idx = 0; idx < nonground_indices.size(); idx++)
        {
            int point_idx = nonground_indices[idx];
            if (point_idx >= 0 && point_idx < tempGroundCloud->size()) 
            {
                cureNonGroundCloud->push_back(tempGroundCloud->points[point_idx]);
            }
        }

        std::vector<std::vector<int>> clusterIndices;

        CVCHandler::Param param;
        CVCHandler cvcHandler(param);
        clusterIndices = cvcHandler.run(cureNonGroundCloud);

        pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZI>(*cureNonGroundCloud));
        // 모든 점의 intensity를 -1로 초기화 (미분류)
        for (auto& point : clusterCloud->points) 
        {
            point.intensity = -1.0f;
        }

        // 각 클러스터에 라벨 할당
        for (int clusterId = 0; clusterId < clusterIndices.size(); ++clusterId) 
        {
            for (int pointIdx : clusterIndices[clusterId]) 
            {
                clusterCloud->points[pointIdx].intensity = static_cast<float>(clusterId);
            }
        }

        pcl::io::savePCDFileBinary(ScansDirectory + to_string(k) + "_ground.pcd", *cureGroundCloud); // scan data
        pcl::io::savePCDFileBinary(ScansDirectory + to_string(k) + "_nonground.pcd", *cureNonGroundCloud); // scan data
        pcl::io::savePCDFileBinary(ScansDirectory + to_string(k) + "_cluster.pcd", *clusterCloud); // scan data

        
        float percent = static_cast<float>(k + 1) / static_cast<float>(total_frames) * 100;
        printf("Ground Segmentation %d/%d completed (%.1f%%)\n", k+1, total_frames, percent);
        fflush(stdout);
    }
    
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_kf (new pcl::KdTreeFLANN<pcl::PointXYZI>());
    std::vector<int> kfSearchInd;
    std::vector<float> kfSearchSqDis;

    pcl::PointCloud<pcl::PointXYZI>::Ptr staticMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr optimizedNGMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr staticNGMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    for (int k = 0; k < total_frames; k++)
    {
        if (g_cancel_save.load()) throw SaveCancelledException{};

        Eigen::Matrix4f TF = createTransformMatrix(keyframePosesUpdated[k]);

        pcl::PointCloud<pcl::PointXYZI>::Ptr CureClusterCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(ScansDirectory + std::to_string(k) + "_cluster.pcd", *CureClusterCloud);
        pcl::transformPointCloud(*CureClusterCloud, *CureClusterCloud, TF);

        std::vector<std::vector<int>> CurClusterIndices;
        std::vector<bool> CurDynamicFlag;
        CurDynamicFlag.assign(CureClusterCloud->points.size(),false);
        
        for (size_t idx = 0; idx < CureClusterCloud->points.size(); idx++)
        {
            if (CureClusterCloud->points[idx].intensity == 0)   
            {
                CurDynamicFlag[idx] = true;
                continue;
            }
            
            int Cluster_idx = CureClusterCloud->points[idx].intensity;
            if (Cluster_idx > CurClusterIndices.size())
            {
                CurClusterIndices.resize(Cluster_idx);  //intensity 0 값들은 패스하므로 1부터 시작
            }
            CurClusterIndices[Cluster_idx-1].push_back(idx);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr subMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
        
        for (int i = keyframePosesUpdated.size()-1; i >= 0; i--)
        {   
            double dist = sqrt(pow(keyframePosesUpdated[k].x-keyframePosesUpdated[i].x,2)
                            +pow(keyframePosesUpdated[k].y-keyframePosesUpdated[i].y,2)
                            +pow(keyframePosesUpdated[k].z-keyframePosesUpdated[i].z,2));
            if (dist > 10)  continue;
            if (i >= k-3 && i <= k+3)  continue;
            Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(keyframePosesUpdated[i].yaw, Eigen::Vector3f::UnitZ())
                    * Eigen::AngleAxisf(keyframePosesUpdated[i].pitch, Eigen::Vector3f::UnitY())
                    * Eigen::AngleAxisf(keyframePosesUpdated[i].roll, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q(rotation);        
            Eigen::Matrix4f curr_TF (Eigen::Matrix4f::Identity());
            curr_TF.block(0,0,3,3) = rotation;
            curr_TF(0,3) = keyframePosesUpdated[i].x;
            curr_TF(1,3) = keyframePosesUpdated[i].y;
            curr_TF(2,3) = keyframePosesUpdated[i].z;
            
            pcl::PointCloud<pcl::PointXYZI>::Ptr KeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::io::loadPCDFile(ScansDirectory + std::to_string(i) + "_nonground.pcd", *KeyframeCloud);
            
            pcl::transformPointCloud(*KeyframeCloud, *KeyframeCloud, curr_TF);
            *subMapCloud += *KeyframeCloud; 
        }

        pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree_submap (new pcl::KdTreeFLANN<pcl::PointXYZI>());
        kdtree_submap->setInputCloud(subMapCloud);
        std::vector<int> closedSearchInd;
        std::vector<float> closedSearchSqDis;
        for (size_t p = 0; p < CurClusterIndices.size(); p++)
        {
            int cnt = 0;
            for (size_t idx = 0; idx < CurClusterIndices[p].size(); idx++)
            {
                kdtree_submap->nearestKSearch(CureClusterCloud->points[CurClusterIndices[p][idx]], 1, closedSearchInd, closedSearchSqDis);
                if (closedSearchSqDis[0] < 0.2)    cnt++;
            }
            int tt = 0;
            double ratio = static_cast<double>(cnt)/ static_cast<double>(CurClusterIndices[p].size());
            if (ratio < 0.95)
            {
                while (tt < CurClusterIndices[p].size())
                {
                    CurDynamicFlag[CurClusterIndices[p][tt]] = true;
                    tt++;
                }
            }
        }
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr removalCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr removalGlobalCloud(new pcl::PointCloud<pcl::PointXYZI>());
        for (int i = 0; i < CureClusterCloud->points.size(); i++)
        {
            if (!CurDynamicFlag[i])  removalGlobalCloud->points.push_back(CureClusterCloud->points[i]);
        }
        pcl::transformPointCloud(*removalGlobalCloud, *removalCloud, TF.inverse());

        *staticNGMapCloud += *removalGlobalCloud;
        *optimizedNGMapCloud += *CureClusterCloud;

        pcl::io::savePCDFileBinary(ScansDirectory + to_string(k) + "_nonground.pcd", *removalCloud); // scan data


        pcl::PointCloud<pcl::PointXYZI>::Ptr GroundCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(ScansDirectory + std::to_string(k) + "_ground.pcd", *GroundCloud);

        *removalCloud += *GroundCloud;
        pcl::io::savePCDFileBinary(ScansDirectory + to_string(k) + ".pcd", *removalCloud); // scan data

        pcl::transformPointCloud(*removalCloud, *removalCloud, TF);
        *staticMapCloud += *removalCloud;

        float percent = static_cast<float>(k + 1) / static_cast<float>(total_frames) * 100;
        printf("Dynamic Object Removal %d/%d completed (%.1f%%)\n", k+1, total_frames, percent);
        fflush(stdout);
    }

    // 최종 다운샘플링
    if (!staticMapCloud->empty()) 
    {
        pcl::VoxelGrid<pcl::PointXYZI> downSizeMapFilter;
        downSizeMapFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);

        downSizeMapFilter.setInputCloud(staticMapCloud);
        downSizeMapFilter.filter(*staticMapCloud);

        downSizeMapFilter.setInputCloud(staticNGMapCloud);
        downSizeMapFilter.filter(*staticNGMapCloud);

        downSizeMapFilter.setInputCloud(optimizedNGMapCloud);
        downSizeMapFilter.filter(*optimizedNGMapCloud);
    }

    printf("Final static map size: %lu points\n", staticMapCloud->size());
    std::cout << "Static map generation completed!" << std::endl;

    pcl::io::savePCDFileBinary(save_directory + "StaticNGMap.pcd", *staticNGMapCloud);
    pcl::io::savePCDFileBinary(save_directory + "OptimizedNGMap.pcd", *optimizedNGMapCloud);
    return staticMapCloud;
}

void save_trajectory_callback(
    const std::shared_ptr<pose_graph_optimization::srv::SaveMap::Request> request,
    std::shared_ptr<pose_graph_optimization::srv::SaveMap::Response> response)
{
    std::string map_directory_name = request->directory_name;
    
    if (map_directory_name.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("posegraphoptimization"), "Directory name cannot be empty");
        response->success = false;
        response->message = "Directory name cannot be empty";
        return;
    }
    
    RCLCPP_INFO(rclcpp::get_logger("posegraphoptimization"), "Saving graph optimization trajectory to directory: %s", map_directory_name.c_str());

    // 새 저장 요청마다 취소 플래그 초기화
    g_cancel_save = false;

    try {
        // 1. 먼저 기존 지도 저장 프로세스 수행 (Map 디렉토리에)

        // Save optimized trajectory to Map directory
        for (int k = 0; k < keyframePosesUpdated.size(); k++) 
        {
            if (g_cancel_save.load()) throw SaveCancelledException{};

            Eigen::Matrix3f rotation;
            rotation = Eigen::AngleAxisf(keyframePosesUpdated[k].yaw, Eigen::Vector3f::UnitZ())
                        * Eigen::AngleAxisf(keyframePosesUpdated[k].pitch, Eigen::Vector3f::UnitY())
                        * Eigen::AngleAxisf(keyframePosesUpdated[k].roll, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q(rotation);
            optimized_stream << keyframeTimes[k] << " " << keyframePosesUpdated[k].x << " " << keyframePosesUpdated[k].y << " " << keyframePosesUpdated[k].z << " " 
                    << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
            
            if (k > 0)
            {
                gtsam::Pose3 poseFrom = Pose6toGTSAMPose3(keyframePosesUpdated[k-1]);
                gtsam::Pose3 poseTo = Pose6toGTSAMPose3(keyframePosesUpdated[k]);
                // odom factor
                gtsam::Pose3 relPose = poseFrom.between(poseTo);
                edge_stream << k-1 << " " << k << " " << relPose.translation().x() << " " << relPose.translation().y() << " " 
                << relPose.translation().z() << " " << relPose.rotation().roll() << " " << relPose.rotation().pitch() << " " << relPose.rotation().yaw() << " " 
                << keyframeCovariance[k-1](0) << " " << keyframeCovariance[k-1](1) << " " << keyframeCovariance[k-1](2) << " "
                << keyframeCovariance[k-1](3) << " " << keyframeCovariance[k-1](4) << " " << keyframeCovariance[k-1](5) << endl;
            }
        }

        // Generate optimized map in Map directory
        if (g_cancel_save.load()) throw SaveCancelledException{};
        response->message = "Generating Optimized Map... ";
        auto optimizedMapCloud = generateOptimizedMap();
        std::string optimized_map_path = save_directory + "OptimizedMap.pcd";
        pcl::io::savePCDFileBinary(optimized_map_path, *optimizedMapCloud);
        optimizedMapCloud.reset();

        // Generate static map with dynamic object removal in Map directory
        if (g_cancel_save.load()) throw SaveCancelledException{};
        pcl::PointCloud<pcl::PointXYZI>::Ptr staticMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
        response->message = "Generating Static Map... ";
        staticMapCloud = generateStaticMap();
        pcl::io::savePCDFileBinary(save_directory + "StaticMap.pcd", *staticMapCloud);

        if (g_cancel_save.load()) throw SaveCancelledException{};
        RCLCPP_INFO(rclcpp::get_logger("posegraphoptimization"), "Map generation completed. Now copying to: %s", map_directory_name.c_str());
        
        // 2. 입력받은 디렉토리 생성
        std::string new_map_directory = string(ROOT_DIR) + map_directory_name + "/";
        auto unused = system((std::string("exec rm -r ") + new_map_directory).c_str());
        unused = system((std::string("mkdir -p ") + new_map_directory).c_str());
        
        // 3. Map 디렉토리 내의 파일들을 새 디렉토리로 복사 (Debug 폴더 제외)
        std::string copy_scans_command = "cp -r " + save_directory + "Scans " + new_map_directory;
        unused = system(copy_scans_command.c_str());
        std::string copy_files_command = "cp " + save_directory + "*.txt " + save_directory + "*.pcd " + new_map_directory + " 2>/dev/null || true";
        unused = system(copy_files_command.c_str());
        
        RCLCPP_INFO(rclcpp::get_logger("posegraphoptimization"), "Map saved to directory: %s", map_directory_name.c_str());
        
        response->success = true;
        response->message = "Map saved successfully to directory: " + map_directory_name;
    }
    catch (const SaveCancelledException&) {
        RCLCPP_INFO(rclcpp::get_logger("posegraphoptimization"), "Save map cancelled by user");
        g_cancel_save = false;
        response->success = false;
        response->message = "Cancelled by user";
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("posegraphoptimization"), "Failed to save map: %s", e.what());
        g_cancel_save = false;
        response->success = false;
        response->message = std::string("Save failed: ") + e.what();
    }
}

void getDirectory()
{
    save_directory = string(ROOT_DIR) + "Map/";
    auto unused = system((std::string("exec rm -r ") + save_directory).c_str());
    unused = system((std::string("mkdir -p ") + save_directory).c_str());

    optimized_poseDirectory = save_directory + "optimized_poses.txt";
    odom_poseDirectory = save_directory + "odom_poses.txt";
    edge_directory = save_directory + "edges.txt";
    odom_stream = std::fstream(odom_poseDirectory, std::fstream::out);
    odom_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!odom_stream) 
    {
        cout<<"Failed to open odom file"<<endl;
    }

    optimized_stream = std::fstream(optimized_poseDirectory, std::fstream::out);
    optimized_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!optimized_stream) 
    {
        cout<<"Failed to open graph optimization file"<<endl;
    }

    edge_stream = std::fstream(edge_directory, std::fstream::out);
    edge_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!edge_stream) 
    {
        cout<<"Failed to open Edge file"<<endl;
    }

    ScansDirectory = save_directory + "Scans/";
    unused = system((std::string("exec rm -r ") + ScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + ScansDirectory).c_str());

    DebugDirectory = save_directory + "Debug/";
    unused = system((std::string("exec rm -r ") + DebugDirectory).c_str());
    unused = system((std::string("mkdir -p ") + DebugDirectory).c_str());

}

void setParams (std::shared_ptr<rclcpp::Node> nh)
{
    patchwork::Params params;

    // Declare parameters with default values and get them directly (no duplication)
    num_scan = nh->declare_parameter("preprocess.scan_line", 16);
    det_range_ = nh->declare_parameter("mapping.det_range", 50.0);
    horizontal_resolution = nh->declare_parameter("preprocess.horizontal_resolution", 1000);
    blind = nh->declare_parameter("preprocess.blind", 0.01);
    R_SOLiD_THRES = nh->declare_parameter("posegraph.r_solid_thres", 0.99);
    FOV_u = nh->declare_parameter("posegraph.fov_u", 2.0);
    FOV_d = nh->declare_parameter("posegraph.fov_d", -24.8);
    NUM_ANGLE = nh->declare_parameter("posegraph.num_angle", 60);
    NUM_RANGE = nh->declare_parameter("posegraph.num_range", 40);
    NUM_HEIGHT = nh->declare_parameter("posegraph.num_height", 32);
    MIN_DISTANCE = nh->declare_parameter("posegraph.min_distance", 3);
    MAX_DISTANCE = nh->declare_parameter("posegraph.max_distance", 80);
    VOXEL_SIZE = nh->declare_parameter("posegraph.voxel_size", 0.4);
    NUM_EXCLUDE_RECENT = nh->declare_parameter("posegraph.num_exclude_recent", 30);
    NUM_CANDIDATES_FROM_TREE = nh->declare_parameter("posegraph.num_candidates_from_tree", 3);
    loop_dist = nh->declare_parameter("posegraph.loop_dist", 10.0);
    dop_thres = nh->declare_parameter("posegraph.dop_thres", 0.5);
    params.sensor_height = nh->declare_parameter("posegraph.sensor_height", 0.0);
    params.num_iter      = nh->declare_parameter("posegraph.num_iter", 3);
    params.num_lpr       = nh->declare_parameter("posegraph.num_lpr", 20);
    params.num_min_pts   = nh->declare_parameter("posegraph.num_min_pts", 0);
    params.th_seeds      = nh->declare_parameter("posegraph.th_seeds", 0.3);

    params.th_dist    = nh->declare_parameter("posegraph.th_dist", 0.125);
    params.th_seeds_v = nh->declare_parameter("posegraph.th_seeds_v", 0.25);
    params.th_dist_v  = nh->declare_parameter("posegraph.th_dist_v", 0.9);

    params.max_range       = nh->declare_parameter("posegraph.max_range", 80.0);
    params.min_range       = nh->declare_parameter("posegraph.min_range", 1.0);
    params.uprightness_thr = nh->declare_parameter("posegraph.uprightness_thr", 0.101);

    params.RNR_ver_angle_thr = nh->declare_parameter("posegraph.ver_angle_thr", -15.0);
    params.enable_RNR = false;

    solidModule.setParams(FOV_u, FOV_d, NUM_ANGLE, NUM_RANGE, NUM_HEIGHT, MIN_DISTANCE, MAX_DISTANCE, VOXEL_SIZE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE, R_SOLiD_THRES);

    // Construct the main Patchwork++ node
    Patchworkpp_ = std::make_unique<patchwork::PatchWorkpp>(params);
    params.num_sectors_each_zone = {16, 16, 27, 16};
    params.uprightness_thr = 0.82;
    Patchworkpp_Fine = std::make_unique<patchwork::PatchWorkpp>(params);

    gicp.setMaxCorrespondenceDistance(10.0);
    gicp.setNumThreads(0);
    gicp.setCorrespondenceRandomness(15);
    gicp.setMaximumIterations(20);
    gicp.setTransformationEpsilon(0.01);
    gicp.setEuclideanFitnessEpsilon(0.01);
    gicp.setRANSACIterations(5);
    gicp.setRANSACOutlierRejectionThreshold(1.0);

    loopLine.type = visualization_msgs::msg::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::msg::Marker::ADD;
    loopLine.color.b = 1.0; loopLine.color.a = 0.5;
    loopLine.scale.x = 0.1;
    loopLine.header.frame_id = "odom";

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

    
}

void SigHandle(int sig)
{
    sig_buffer.notify_all();
    rclcpp::shutdown(); // ROS 종료
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("posegraphoptimization");

    // QoS for visualization topics (latest data only)
    auto qos_viz = rclcpp::QoS(rclcpp::KeepLast(1));
    qos_viz.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_viz.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // QoS for SLAM data processing (buffered, no data loss)
    auto qos_slam = rclcpp::QoS(rclcpp::KeepLast(10));  // 모든 메시지 보관
    qos_slam.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);  // 메시지 손실 없음
    qos_slam.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    getDirectory();

    setParams(nh);

    initNoises();

    // SLAM 데이터는 손실 없이 버퍼링
    auto sub_keyframe = nh->create_subscription<fast_lio::msg::Frame>("/key_frame", qos_slam, kf_callback);
    
    // 시각화 데이터는 최신 데이터만
    kf_node_pub = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/kf_node", qos_viz);
    LoopLineMarker_pub = nh->create_publisher<visualization_msgs::msg::Marker>("/loopLine", qos_viz);
    PubPGO_path = nh->create_publisher<nav_msgs::msg::Path>("/PGO_path", qos_viz);
    PubPGO_map = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/PGO_map", qos_viz);
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    signal(SIGINT, SigHandle);
    std::thread lc_detection {process_lcd}; // loop closure detection 
    std::thread edge_calculation {process_edge};    //GICP based edge measurement calculation
    std::thread graph_optimization {process_optimization};  //pose graph optimization
    std::thread vis_map {process_viz};  //Map Visualization

    // 저장 서비스 생성
    save_service_ = nh->create_service<pose_graph_optimization::srv::SaveMap>(
        "save_trajectory", 
        std::bind(&save_trajectory_callback, std::placeholders::_1, std::placeholders::_2)
    );

    // 저장 취소 서비스 생성
    cancel_service_ = nh->create_service<std_srvs::srv::Trigger>(
        "cancel_save_trajectory",
        std::bind(&cancel_save_callback, std::placeholders::_1, std::placeholders::_2)
    );

    rclcpp::spin(nh);
    rclcpp::shutdown();

    odom_stream.close();
    optimized_stream.close(); 
    edge_stream.close(); 

    return 0;
}