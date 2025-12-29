#ifndef NAVOCR_PROJECTION__NAVOCR_SLAM_NODE_HPP_
#define NAVOCR_PROJECTION__NAVOCR_SLAM_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <memory>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <deque>
#include <algorithm>

namespace navocr_projection
{

struct Observation
{
  rclcpp::Time timestamp;
  Eigen::Vector3d camera_position;
  Eigen::Quaterniond camera_orientation;
  double bbox_center_u;
  double bbox_center_v;
  int bbox_size_x;
  int bbox_size_y;
};

struct Landmark
{
  Eigen::Vector3d mean_position;       // Mean position (μ)
  Eigen::Matrix3d covariance;          // Covariance matrix (Σ)
  int observation_count;               // Number of observations
  std::deque<std::pair<std::string, double>> text_history; // Recent OCR texts with confidence (text, conf)
  std::string representative_text;     // Most common text (weighted by confidence)
  double text_confidence;              // Confidence in text (0-1)
  rclcpp::Time last_updated;           // Last observation time
  int landmark_id;                     // Unique ID

  std::vector<Observation> observations;  // Store all observations for reprojection error

  Landmark()
    : mean_position(Eigen::Vector3d::Zero()),
      covariance(Eigen::Matrix3d::Identity()),
      observation_count(0),
      text_confidence(0.0),
      landmark_id(-1) {}
};

class NavOCRSLAMNode : public rclcpp::Node
{
public:
  explicit NavOCRSLAMNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~NavOCRSLAMNode();

private:
  void detectionCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr depth_msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg);
  
  Eigen::Vector3d projectTo3D(double pixel_u, double pixel_v, double depth_m);
  bool transformToWorld(const Eigen::Vector3d & point_in_camera_frame, const rclcpp::Time & timestamp, 
                        Eigen::Vector3d & point_in_world_frame);
  void publishMarkers();
  
  void addObservationToLandmarks(const Eigen::Vector3d & world_pos, const std::string & text,
                                  const rclcpp::Time & timestamp, double ocr_confidence,
                                  double bbox_center_u, double bbox_center_v,
                                  int bbox_size_x, int bbox_size_y);
  double mahalanobisDistance(const Eigen::Vector3d & point, const Landmark & landmark);
  double levenshteinDistance(const std::string & s1, const std::string & s2);
  double textSimilarity(const std::string & s1, const std::string & s2);
  void updateLandmark(Landmark & landmark, const Eigen::Vector3d & new_pos, const std::string & new_text,
                      const rclcpp::Time & timestamp, double ocr_confidence,
                      double bbox_center_u, double bbox_center_v,
                      int bbox_size_x, int bbox_size_y);
  void createLandmark(const Eigen::Vector3d & pos, const std::string & text, const rclcpp::Time & timestamp,
                      double ocr_confidence);
  void updateRepresentativeText(Landmark & landmark);
  void checkAndMergeNewLandmark(size_t new_idx);
  Eigen::Vector3d getCurrentRobotPosition();
  void saveLandmarks();

  // Reprojection error calculation
  void computeReprojectionErrors();
  void saveReprojectionSummary();
  
  // Helper function for marker visualization
  std_msgs::msg::ColorRGBA getConfidenceColor(double confidence) const;
  
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  
  std::string output_dir_;
  double confidence_threshold_;
  std::string camera_frame_;
  std::string world_frame_;
  
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_;
  double fx_, fy_, cx_, cy_;
  bool camera_info_received_;
  
  std::map<rclcpp::Time, cv::Mat> depth_buffer_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
  const size_t buffer_size_ = 50;
  
  std::vector<Landmark> landmarks_;
  int next_landmark_id_;
  
  double sensor_noise_std_;
  int min_observations_;
  double merge_search_radius_;
  
  const double chi2_threshold_;
  const double text_similarity_threshold_;
  const double acceptance_threshold_;
  const int text_history_size_;
  
  // Filtering thresholds (used for both visualization and saving)
  const double min_confidence_for_output_;  // Minimum confidence for RViz & CSV
  
  // Visualization parameters (fixed at compile time)
  const double marker_cube_size_;
  const double marker_transparency_;
  const double text_marker_height_;
  const double confidence_threshold_high_;
  const double confidence_threshold_medium_;
  
  template<typename T>
  typename std::map<rclcpp::Time, T>::iterator findClosestTimestamp(
    std::map<rclcpp::Time, T>& buffer, const rclcpp::Time& target_time)
  {
    if (buffer.empty()) return buffer.end();
    auto it = buffer.lower_bound(target_time);
    if (it == buffer.end()) return std::prev(buffer.end());
    if (it == buffer.begin()) return it;
    auto prev_it = std::prev(it);
    auto dt1 = std::abs((target_time - prev_it->first).nanoseconds());
    auto dt2 = std::abs((it->first - target_time).nanoseconds());
    return (dt1 < dt2) ? prev_it : it;
  }
};

}

#endif
