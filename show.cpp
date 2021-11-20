// based on https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/

#include <filesystem>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::filesystem::path query_image_filepath { "query.png" };
std::filesystem::path train_image_filepath { "train.jpg" };

constexpr size_t num_top_matches = 30;

int main()
{
    cv::Mat query_image = cv::imread(query_image_filepath);
    cv::Mat train_image = cv::imread(train_image_filepath);

    cv::Mat query_image_gs, train_image_gs;
    cv::cvtColor(query_image, query_image_gs, cv::COLOR_BGR2GRAY);
    cv::cvtColor(train_image, train_image_gs, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat query_image_gs_gpu, train_image_gs_gpu;
    query_image_gs_gpu.upload(query_image_gs);
    train_image_gs_gpu.upload(train_image_gs);

    auto orb = cv::cuda::ORB::create();
    std::vector<cv::KeyPoint> query_keypoints, train_keypoints;
    cv::cuda::GpuMat query_descriptors_gpu, train_descriptors_gpu;
    orb->detectAndCompute(query_image_gs_gpu, cv::cuda::GpuMat(), query_keypoints, query_descriptors_gpu);
    orb->detectAndCompute(train_image_gs_gpu, cv::cuda::GpuMat(), train_keypoints, train_descriptors_gpu);

    cv::Mat query_descriptors, train_descriptors;
    query_descriptors_gpu.download(query_descriptors);
    train_descriptors_gpu.download(train_descriptors);
    auto matcher = cv::BFMatcher();
    std::vector<cv::DMatch> matches;
    matcher.match(query_descriptors, train_descriptors, matches);
    std::sort(matches.begin(), matches.end());

    size_t actual_num_top_matches = std::min(num_top_matches, matches.size());
    std::vector<cv::DMatch> top_matches { actual_num_top_matches };
    std::copy(matches.cbegin(), matches.cbegin() + actual_num_top_matches, top_matches.begin());

    cv::Mat final_image;
    cv::drawMatches(query_image, query_keypoints, train_image, train_keypoints, top_matches, final_image);

    cv::namedWindow("final", cv::WINDOW_KEEPRATIO);
    cv::imshow("final", final_image);
    cv::waitKey(0);
}
