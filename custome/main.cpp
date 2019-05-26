#include <iostream>

#include <memory>
#include <chrono>
#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "darknet.h"

using std::cout;
using std::endl;
using std::string;
using namespace cv;


IplImage *image_to_ipl(image im)
{
    int x,y,c;
    IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for(y = 0; y < im.h; ++y){
        for(x = 0; x < im.w; ++x){
            for(c= 0; c < im.c; ++c){
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
            }
        }
    }
    return disp;
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c); 
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }   
    return im; 
}

Mat image_to_mat(image im) 
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    Mat m = cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

struct tracker_node {
    Ptr<Tracker> tracker;
    int fail_count = 0;
    Scalar color;
};

void detect_video(char *cfgfile, char *weightfile, string filename, char *datacfg, string output="out.avi") {
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    const float class_thresh = .4;
    const float box_thresh = .4;

    image **alphabet = load_alphabet();

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    cv::Mat mat;
    cv::VideoCapture video(filename);
    cout << filename << endl;
    int video_width = int(video.get(CV_CAP_PROP_FRAME_WIDTH)),
        video_height = int(video.get(CV_CAP_PROP_FRAME_HEIGHT)),
        codec = video.get(CV_CAP_PROP_FOURCC);

    float video_fps = video.get(CV_CAP_PROP_FPS);
    // cout << video_fps << endl;

    cv::VideoWriter out(output, codec, video_fps, cv::Size(net->w, net->h), true);
    layer l = net->layers[net->n-1];
    if(!video.isOpened()) {
        cout << "can't open video" << endl;
        exit(1);
    }

    make_window("predictions", 512, 512, 0);
    int missing_box_count = 0;
    Scalar rectangle_color;
    std::list<tracker_node> tracker_list;
    while(video.isOpened()) {
        auto start_time = std::chrono::steady_clock::now();
        video >> mat;
        cv::Mat dst;
        if(mat.empty()) break;
        cv::resize(std::move(mat), dst, Size(net->w, net->h));
        image im = mat_to_image(dst);
        network_predict(net, im.data);
        int nboxes;
        detection *dets = get_network_boxes(net, im.w, im.h, .4, .4, 0, 1, &nboxes);
        do_nms_sort(dets, nboxes, l.classes, box_thresh);


        std::list<detection*> unuse_list;
        for(int i = 0; i < nboxes; ++i) {
            const detection &det = dets[i];
            if(det.prob[1] > class_thresh) { // prob[1] is bicycle
                unuse_list.push_back(dets + i);
                break;
            }
        }
        for(auto iterator = tracker_list.begin();
                iterator != tracker_list.end();) {
            const Ptr<Tracker> &tracker = iterator->tracker;
            Rect2d rect;
            if(!tracker->update(dst, rect)) {
                iterator->fail_count += 1;
                if(iterator->fail_count > (int)video_fps * 2){
                    tracker_list.erase(iterator++);
                    continue;
                }
                ++iterator;
                continue;
            } else {
                rectangle(dst, rect, iterator->color, 2, 1);
            }
            box b;
            b.w = rect.width / net->w;
            b.h = rect.height / net->h;
            b.x = rect.x / net->w + b.w / 2;
            b.y = rect.y / net->h + b.h / 2;
            for(auto iter_det = unuse_list.begin(); iter_det != unuse_list.end(); ++iter_det) {
                const detection &det = **iter_det;
                if(box_iou(b, det.bbox) > box_thresh) {
                    unuse_list.erase(iter_det);
                    iterator->fail_count = 0;
                    break;
                }
            }
            ++iterator;
        }

        for(auto iter_det = unuse_list.begin(); iter_det != unuse_list.end(); ++iter_det) {
            const box &bbox = (*iter_det)->bbox;
            Rect2d rect;
            rect.width = bbox.w * net->w;
            rect.height = bbox.h * net->h;
            rect.x = (bbox.x - bbox.w / 2) * net->w;
            rect.y = (bbox.y - bbox.h / 2) * net->h;
            tracker_node node;
            node.tracker = TrackerKCF::create();
            node.tracker->init(dst, rect);
            node.color = Scalar(rand()%255, rand()%255, rand()%255);
            tracker_list.push_back(node);
            rectangle(dst, rect, node.color, 2, 1);
        }

        draw_detections(im, dets, nboxes, .4, names, alphabet, l.classes);
        free_detections(dets, nboxes);

        mat = std::move(image_to_mat(im));
        cv::imshow("tracing", dst);
        cv::waitKey(1);
        out.write(dst);
        cv::imshow(std::string("predictions"), mat);
        cv::waitKey(1);
        free_image(im);

        auto end_time = std::chrono::steady_clock::now();

        cout << "use: " << std::chrono::duration_cast<std::chrono::milliseconds >(end_time - start_time).count() << " milliseconds " << endl;
    }
}

// void test_detector(char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh)
// {
//     char datacfg[] = "../cfg/coco.data";
//     list *options = read_data_cfg(datacfg);
//     char *name_list = option_find_str(options, "names", "data/names.list");
//     char **names = get_labels(name_list);
// 
//     image **alphabet = load_alphabet();
//     network *net = load_network(cfgfile, weightfile, 0);
//     set_batch_network(net, 1);
//     double time;
//     char buff[256];
//     char *input = buff;
//     float nms=.45;
//     while(1){
//         if(filename){
//             strncpy(input, filename, 256);
//         } else {
//             printf("Enter Image Path: ");
//             fflush(stdout);
//             input = fgets(input, 256, stdin);
//             if(!input) return;
//             strtok(input, "\n");
//         }
//         image im = load_image_color(input,0,0);
//         image sized = letterbox_image(im, net->w, net->h);
//         //image sized = resize_image(im, net->w, net->h);
//         //image sized2 = resize_max(im, net->w);
//         //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
//         //resize_network(net, sized.w, sized.h);
//         layer l = net->layers[net->n-1];
// 
// 
//         float *X = sized.data;
//         time=what_time_is_it_now();
//         network_predict(net, X);
//         printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
//         int nboxes = 0;
//         detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
// 
//         //printf("%d\n", nboxes);
//         //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
//         if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
//         for(int i = 0; i < nboxes; ++i) {
//             int index = 0;
//             float max = -1;
//             for(int o = 0; o < l.classes; ++o) {
//                 if(max < dets[i].prob[o]) {
//                     max = dets[i].prob[o];
//                     index = o;
//                 }
//             }
//             cout << names[index] << ": " << max << endl;
//         }
//         draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
//         free_detections(dets, nboxes);
//         save_image(im, "predictions");
//         make_window("predictions", 512, 512, 0);
//         show_image(im, "predictions", 0);
// 
//         free_image(im);
//         free_image(sized);
//         if (filename) break;
//     }
// }

int main(int argc, char **argv)
{
    if(argc != 5) {
        cout << "usage: " << argv[0] << " <cfg> <weights> <input video> <datacfg>" << endl;
        exit(1);
    }
    detect_video(argv[1], argv[2], argv[3], argv[4]);
    return 0;
}
