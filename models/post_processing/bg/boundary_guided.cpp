#include <queue>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <map>
#include <sys/time.h>
#include <unordered_map>
#include <thread>
#include <future>
#include <tuple>
#include "clipper.hpp"  

using namespace ClipperLib;
using namespace std;
using namespace cv;

namespace py = pybind11;

namespace BG{
    void get_roi(const std::vector<cv::Point> contour, 
                 int height, int width, cv::Rect &roi){
        roi = cv::boundingRect(contour);
        int minX = roi.x;
        int minY = roi.y;
        int maxX = roi.x + roi.width;
        int maxY = roi.y + roi.height;
        int appendVal = 10;
        
        minX -= appendVal;
        minY -= appendVal;
        maxX += appendVal;
        maxY += appendVal;
        
        minX = minX < 0 ? 0:minX;
        minY = minY < 0 ? 0:minY;
        maxX = maxX > (width-1) ? (width-1):maxX;
        maxY = maxY > (height-1) ? (height-1):maxY;
        
        roi.x = minX;
        roi.y = minY;
        roi.width = maxX - minX;
        roi.height = maxY - minY;
    }

    void get_areascore(const cv::Mat matlabel_map, const cv::Mat scoremap, 
                          int pixelVal, int height, int width,
                          int &area, float &meanscore){
        area = 0;
        std::vector<float> scores;
        for(int r=0; r<height; r++){
            const uchar *labeldata_row = matlabel_map.ptr<uchar>(r);
            const float *score_row = scoremap.ptr<float>(r);
            for(int c=0; c<width; c++){
                if(labeldata_row[c] == pixelVal){
                    area += 1;
                    scores.emplace_back(score_row[c]);
                }
            }
        }
        meanscore = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    }
    
    void unclip_dist(std::vector<cv::Point> polys, float dist, 
                     std::vector< std::vector<cv::Point> > &expand){
        Path subj;
        Paths solution;
        for(int i=0; i<polys.size(); i++){
            subj << IntPoint(polys[i].x, polys[i].y);
        }
        ClipperOffset co;
        co.AddPath(subj, jtRound, etClosedPolygon);
        co.Execute(solution, dist);
        for (int i=0; i<solution.size(); i++){
            std::vector<cv::Point> inner;
            for (int j=0; j<solution[i].size(); j++){
                inner.emplace_back(cv::Point(solution[i][j].X, solution[i][j].Y));
            }
            expand.emplace_back(inner);
        }
    }
    
    void gen_roiinfo(int height, int width, cv::Rect roi, cv::Rect &retroi){
        int minX = roi.x;
        int minY = roi.y;
        int maxX = roi.x + roi.width;
        int maxY = roi.y + roi.height;
        int appendVal = 10;
        
        minX -= appendVal;
        minY -= appendVal;
        maxX += appendVal;
        maxY += appendVal;
        
        minX = minX < 0 ? 0:minX;
        minY = minY < 0 ? 0:minY;
        maxX = maxX > (width-1) ? (width-1):maxX;
        maxY = maxY > (height-1) ? (height-1):maxY;
        
        retroi.x = minX;
        retroi.y = minY;
        retroi.width = maxX - minX;
        retroi.height = maxY - minY;
    }
    
    std::tuple<std::vector<std::vector<int32_t>>, std::vector<float>, double >
              bg_ccl(py::array_t<float, py::array::c_style> score,
                     py::array_t<uint8_t, py::array::c_style> textMask,
                     py::array_t<uint8_t, py::array::c_style> kernelMap,
                     py::array_t<float, py::array::c_style> distMap,
                     float min_score, int min_area, int scale){
        struct timeval t0, t1, t2, t3, t4, t5, t6;
        double post_time = 0.0;
        double inner_loop = 0.0;
        gettimeofday(&t0, NULL);
        std::tuple<std::vector<std::vector<int32_t>>, std::vector<float> > results;
        std::vector<std::vector<int32_t>> expand_bbox;
        std::vector<float> bbox_score;
        // gettimeofday(&t0, NULL);
        auto pbuf_score = score.request();
        auto pbuf_textmask = textMask.request();
        auto pbuf_kernelmap = kernelMap.request();
        auto pbuf_distmap = distMap.request();
        
        auto ptr_score = static_cast<float *>(pbuf_score.ptr);
        auto ptr_textmask = static_cast<uint8_t *>(pbuf_textmask.ptr);      
        auto ptr_kernelmap = static_cast<uint8_t *>(pbuf_kernelmap.ptr);
        auto ptr_distmap = static_cast<float *>(pbuf_distmap.ptr);  
        
        auto data_shape = pbuf_score.shape;    // C*H*W
        int h = data_shape[0];  // height
        int w = data_shape[1];  // width
        // kernel shape
        int kh = pbuf_kernelmap.shape[0];  // height
        int kw = pbuf_kernelmap.shape[1];  // width
        //int c = data_shape[0];
        // std::cout << h << ", " << w << std::endl;
        
        if (pbuf_score.ndim != 2 || pbuf_score.shape[0]==0 || pbuf_score.shape[1]==0)
           throw std::runtime_error("pbuf_score must have a shape of (h>0, w>0)");
        
        if (pbuf_textmask.ndim != 2 || pbuf_textmask.shape[0]==0 || pbuf_textmask.shape[1]==0)
           throw std::runtime_error("pbuf_textmask must have a shape of (h>0, w>0)");
        
        if (pbuf_kernelmap.ndim != 2 || pbuf_kernelmap.shape[0]==0 || pbuf_kernelmap.shape[1]==0)
           throw std::runtime_error("pbuf_kernelmap must have a shape of (h>0, w>0)");
        
        if (pbuf_distmap.ndim != 2 || pbuf_distmap.shape[0]==0 || pbuf_distmap.shape[1]==0)
           throw std::runtime_error("pbuf_distmap must have a shape of (h>0, w>0)");
        
        cv::Mat scoremap = cv::Mat::zeros(h, w, CV_32F);
        cv::Mat textmask = cv::Mat::zeros(h, w, CV_8U);
        cv::Mat kernelmap = cv::Mat::zeros(h, w, CV_8U);
        cv::Mat distmap = cv::Mat::zeros(h, w, CV_32F);

        for (int i=0; i<kh; i++){
            for (int j=0; j<kw; j++){
                kernelmap.at<uint8_t>(i, j) = ptr_kernelmap[i*kw + j]*255;
            }
        }
        for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){
                scoremap.at<float>(i, j) = ptr_score[i*w + j];
                textmask.at<uint8_t>(i, j) = ptr_textmask[i*w + j];
                //kernelmap.at<uint8_t>(i, j) = ptr_kernelmap[i*w + j]*255;
                distmap.at<float>(i, j) = ptr_distmap[i*w + j];
            }
        }
        cv::Mat matlabel_map(kernelmap.size(), CV_32S);
        // int label_num = cv::connectedComponents(kernelmap, matlabel_map, 4);
        cv::Mat stats, centroids;
        int label_num = cv::connectedComponentsWithStats(kernelmap, matlabel_map, stats, centroids, 4);
        // matlabel_map.convertTo(matlabel_map, CV_32F);
        gettimeofday(&t1, NULL);
        // could be multi thread        
        for(int i=1; i<label_num; i++){
            gettimeofday(&t3, NULL);

            int con_x = stats.at<int>(cv::Point(0, i));
            int con_y = stats.at<int>(cv::Point(1, i));
            int con_w = stats.at<int>(cv::Point(2, i));
            int con_h = stats.at<int>(cv::Point(3, i));
            cv::Rect roi(con_x, con_y, con_w, con_h);  // will be changed
            cv::Rect appedRoi;
            gen_roiinfo(h, w, roi, appedRoi);
            cv::Mat roi_label = matlabel_map(appedRoi).clone();
            int area = 0; float meanscore = 0.0;
            cv::Mat1b mask_i = roi_label == i;
            std::vector< std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i > hierarchy;
            cv::findContours(mask_i, contours, hierarchy, CV_RETR_EXTERNAL, 
                             CV_CHAIN_APPROX_SIMPLE, cv::Point(appedRoi.x, appedRoi.y));
            
            if(contours.size() <= 0) continue;
            std::vector<cv::Point> offset_contour;
            for (int j=0; j<contours[0].size(); j++){
                contours[0][j].x *= scale;
                contours[0][j].y *= scale;
                offset_contour.push_back(contours[0][j]);
            }
            
            float mean_dist = 0.0;
            std::vector<float> truedist;
            for(int i=0; i<offset_contour.size(); i++){
                float disti = distmap.at<float>(offset_contour[i].y, offset_contour[i].x);
                if(disti > 0){
                    truedist.emplace_back(disti);
                }
            }
            if (truedist.size()==0) continue;
            mean_dist = std::accumulate(truedist.begin(), truedist.end(), 0.0) / truedist.size();
            gettimeofday(&t4, NULL);
            std::vector< std::vector<cv::Point> > expand_poly;

            // printf("expand distance: %u\n", mean_dist);
            unclip_dist(contours[0], mean_dist, expand_poly);
            if (expand_poly.size() == 0) continue;

            cv::Rect new_roi;
            get_roi(expand_poly[0], h, w, new_roi);
            cv::Mat textline = cv::Mat::zeros(new_roi.height, new_roi.width, CV_8U);
            cv::Mat truetext = cv::Mat::zeros(new_roi.height, new_roi.width, CV_8U);
            cv::Mat roi_mask = textmask(new_roi).clone();
            // printf("expand poly num: %u\n", expand_poly.size());
            
            cv::drawContours(textline, expand_poly, 0, cv::Scalar(255), -1, 8, std::vector<cv::Vec4i>(), 0, cv::Point(-new_roi.x, -new_roi.y) );
            truetext = textline.mul(roi_mask);
            gettimeofday(&t5, NULL);
            inner_loop += ((t5.tv_sec - t3.tv_sec) * 1000000 + (t5.tv_usec - t3.tv_usec))/1000.0;
            int leftx = 0, lefty = 0;
            // std::cout << "debug roi_findcontoursv2 0" << std::endl;
            // std::cout << "truetext shape " << truetext.rows << ", " << truetext.cols << std::endl;
            cv::Mat roi_score = scoremap(new_roi).clone();
            get_areascore(truetext, roi_score, 255, new_roi.height, new_roi.width, area, meanscore);  // sum and multiply&sum
            // printf("area: %u, score: %f\n", area, meanscore);
            if (area < min_area) continue;
            if (meanscore < min_score) continue;
            std::vector< std::vector<cv::Point> > contours_bin;
            std::vector<cv::Vec4i > hierarchy_bin;
            cv::findContours(truetext, contours_bin, hierarchy_bin, CV_RETR_EXTERNAL, 
                             CV_CHAIN_APPROX_SIMPLE, cv::Point(new_roi.x, new_roi.y));
            // std::cout << "debug roi_findcontoursv2 1" << std::endl;
            int max_nums = 0;
            int max_index = 0;
            
            for (int i=0; i<contours_bin.size(); i++){
                if (contours_bin[i].size() > max_nums){
                    max_nums = contours_bin[i].size();
                    max_index = i;
                }
            }
            std::vector<int32_t> restmp;
            for(int kk = 0; kk < contours_bin[max_index].size(); kk++){
                restmp.emplace_back(contours_bin[max_index][kk].x);
                restmp.emplace_back(contours_bin[max_index][kk].y);
            }
            expand_bbox.emplace_back(restmp);
            bbox_score.emplace_back(meanscore);
            gettimeofday(&t6, NULL);
            //printf("tcem inner process TIME STAT us: s1: (%d), s2: (%d), s3: (%d), all: [%d] \n", 
            //    ((t4.tv_sec - t3.tv_sec) * 1000000 + (t4.tv_usec - t3.tv_usec)),
            //    ((t5.tv_sec - t4.tv_sec) * 1000000 + (t5.tv_usec - t4.tv_usec)),
            //    ((t6.tv_sec - t5.tv_sec) * 1000000 + (t6.tv_usec - t5.tv_usec)),
            //    ((t6.tv_sec - t3.tv_sec) * 1000000 + (t6.tv_usec - t3.tv_usec)));
        }
        gettimeofday(&t2, NULL);
        post_time = inner_loop + ((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec))/1000.0;
        //printf("tcem post process TIME STAT us: s1: (%d), s2: (%d), all: [%d] \n", 
        //            ((t1.tv_sec - t0.tv_sec) * 1000000 + (t1.tv_usec - t0.tv_usec)),
        //            ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec)),
        //            ((t2.tv_sec - t0.tv_sec) * 1000000 + (t2.tv_usec - t0.tv_usec)));
        return std::make_tuple(expand_bbox, bbox_score, post_time);  // change format
    }

}

PYBIND11_MODULE(bg, m){
    m.def("boundary_guided", &BG::bg_ccl, " re-implementation boundary guided algorithm(cpp)", 
          py::arg("score"), py::arg("textMask"), py::arg("kernelMap"), 
          py::arg("distMap"),py::arg("min_score"), py::arg("min_area"), py::arg("scale") );
}
