// based on https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/

#include <filesystem>

#include <benchmark/benchmark.h>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

std::filesystem::path query_image_filepath { "query.png" };
std::filesystem::path train_image_filepath { "train.jpg" };

static std::pair<cv::Mat, cv::Mat> get_images() {
    cv::Mat query_image = cv::imread(query_image_filepath);
    cv::Mat train_image = cv::imread(train_image_filepath);

    cv::Mat query_image_gs, train_image_gs;
    cv::cvtColor(query_image, query_image_gs, cv::COLOR_BGR2GRAY);
    cv::cvtColor(train_image, train_image_gs, cv::COLOR_BGR2GRAY);

    return { query_image_gs, train_image_gs };
}

static void BM_ORB_query_CPU(benchmark::State& state)
{
    auto pair = get_images();
    const cv::Mat& query_image_gs = pair.first;
    const cv::Mat& train_image_gs = pair.second;

    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> query_keypoints, train_keypoints;
    cv::Mat query_descriptors, train_descriptors;

    for (auto _ : state)
    {
        orb->detectAndCompute(query_image_gs, cv::Mat(), query_keypoints, query_descriptors);
    }
}

static void BM_ORB_train_CPU(benchmark::State& state)
{
    auto pair = get_images();
    const cv::Mat& query_image_gs = pair.first;
    const cv::Mat& train_image_gs = pair.second;

    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> query_keypoints, train_keypoints;
    cv::Mat query_descriptors, train_descriptors;

    orb->detectAndCompute(query_image_gs, cv::Mat(), query_keypoints, query_descriptors);
    
    for (auto _ : state)
    {
        orb->detectAndCompute(train_image_gs, cv::Mat(), train_keypoints, train_descriptors);
    }
}

static void BM_ORB_query_CUDA(benchmark::State& state)
{
    auto pair = get_images();
    const cv::Mat& query_image_gs = pair.first;
    const cv::Mat& train_image_gs = pair.second;

    cv::cuda::GpuMat query_image_gs_gpu, train_image_gs_gpu;
    query_image_gs_gpu.upload(query_image_gs);
    train_image_gs_gpu.upload(train_image_gs);

    auto orb = cv::cuda::ORB::create();
    std::vector<cv::KeyPoint> query_keypoints, train_keypoints;
    cv::cuda::GpuMat query_descriptors_gpu, train_descriptors_gpu;

    for (auto _ : state)
    {
        orb->detectAndCompute(query_image_gs_gpu, cv::cuda::GpuMat(), query_keypoints, query_descriptors_gpu);
    }
}

static void BM_ORB_train_CUDA(benchmark::State& state)
{
    auto pair = get_images();
    const cv::Mat& query_image_gs = pair.first;
    const cv::Mat& train_image_gs = pair.second;

    cv::cuda::GpuMat query_image_gs_gpu, train_image_gs_gpu;
    query_image_gs_gpu.upload(query_image_gs);
    train_image_gs_gpu.upload(train_image_gs);

    auto orb = cv::cuda::ORB::create();
    std::vector<cv::KeyPoint> query_keypoints, train_keypoints;
    cv::cuda::GpuMat query_descriptors_gpu, train_descriptors_gpu;

    orb->detectAndCompute(query_image_gs_gpu, cv::cuda::GpuMat(), query_keypoints, query_descriptors_gpu);
    
    for (auto _ : state)
    {
        orb->detectAndCompute(train_image_gs_gpu, cv::cuda::GpuMat(), train_keypoints, train_descriptors_gpu);
    }
}

static void BM_match_descriptors(benchmark::State& state) {
    auto pair = get_images();
    const cv::Mat& query_image_gs = pair.first;
    const cv::Mat& train_image_gs = pair.second;

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

    for (auto _ : state)
    {
        matcher.match(query_descriptors, train_descriptors, matches);
        matches.clear();
    }
}

BENCHMARK(BM_ORB_query_CPU);
BENCHMARK(BM_ORB_train_CPU);
BENCHMARK(BM_ORB_query_CUDA);
BENCHMARK(BM_ORB_train_CUDA);
BENCHMARK(BM_match_descriptors);

BENCHMARK_MAIN();
