#include <iostream>

#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <csignal>

#include "darknet.h"

using std::cout;
using std::endl;
using std::string;

#define BUFFER_SIZE 416*416*3

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

cv::Mat image_to_mat(image im) 
{
    image copy = copy_image(im);
    constrain_image(copy);
    if(im.c == 3) rgbgr_image(copy);

    IplImage *ipl = image_to_ipl(copy);
    cv::Mat m = cv::cvarrToMat(ipl, true);
    cvReleaseImage(&ipl);
    free_image(copy);
    return m;
}

image mat_to_image(cv::Mat m)
{
    IplImage ipl = m;
    image im = ipl_to_image(&ipl);
    rgbgr_image(im);
    return im;
}

int server_fd;
void detect_video(char *cfgfile, char *weightfile, int port=14333) {
    struct sockaddr_in address;
    try {
        std::size_t sizeof_size;
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if(server_fd == 0) {
            cout << "Open socket error" << endl;
            exit(1);
        }
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        address.sin_addr.s_addr = INADDR_ANY;

        sizeof_size = sizeof(address);
        int result = bind(server_fd, (struct sockaddr *)&address, sizeof_size);
        if (result < 0) {
            cout << "bind server error" << endl;
            exit(result);
        }
        if(listen(server_fd, 3) < 0) {
            cout << "listen error" << endl;
            exit(1);
        }
        int socket;
        unsigned char buffer[BUFFER_SIZE] = { 0 };

        network *net = load_network(cfgfile, weightfile, 0);
        set_batch_network(net, 1);
        layer l = net->layers[net->n-1];

        sizeof_size = sizeof(address);
        while(socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&sizeof_size)) {
            cout << "accept socket" << endl;
            const float box_thresh = .4;
            const float class_thresh = .1;
            if (socket < 0) {
                cout << "accept socket error" << endl;
                exit(socket);
            }
            cout << "perpare read image" << endl;
            int offset = read(socket, (void *)buffer, BUFFER_SIZE);
            size_t image_size = *(size_t*)buffer;
            cout << "read image size " << image_size << endl;
            if(image_size >=  BUFFER_SIZE) {
                cout << "image size too large" << endl;
                close(socket);
                continue;
            }
            while(offset < image_size) {
                offset += read(socket, (void *)(buffer + offset), BUFFER_SIZE - offset);
            }
            if(offset - sizeof(size_t) != image_size) {
                cout << "image size not match" << endl;
                close(socket);
                continue;
            }
            cv::Mat mat;
            std::vector<unsigned char> data(buffer + sizeof(size_t), buffer + offset);
            cv::imdecode(data, cv::IMREAD_UNCHANGED, &mat);
            cv::Mat dst;
            if(mat.empty()) break;
            cv::resize(std::move(mat), dst, cv::Size(net->w, net->h));
            image im = mat_to_image(dst);
            cout << "detected" << endl;
            network_predict(net, im.data);

            int nboxes;
            detection *dets = get_network_boxes(net, im.w, im.h, .2, .2, 0, 1, &nboxes);
            cout << "get " << nboxes << " box" << endl;
            do_nms_sort(dets, nboxes, l.classes, box_thresh);

            std::vector<detection*> valid_class;

            for(int i = 0; i < nboxes; ++i) {
                const detection &det = dets[i];
                int max_class = 0;
                float max_prob = 0;
                for(int c = 0; c < l.classes; ++c) {
                    if(det.prob[c] > max_prob) {
                        max_prob = det.prob[c];
                        max_class = c;
                    }
                }
                if(max_prob < class_thresh) {
                    cout << max_prob << " " << class_thresh << endl;
                    continue;
                }
                valid_class.push_back(dets + i);
            }

            auto valid_size = valid_class.size();
            cout << valid_class.size() << endl;
            size_t send_buffer_size = sizeof(float) * 4 * valid_class.size();
            unsigned char *send_buffer = new unsigned char[send_buffer_size];
            offset = 0;
            for(auto iter = valid_class.begin(); iter != valid_class.end(); ++iter) {
                const auto &det = **iter;
                const box &b = det.bbox;
                (*((float *)(send_buffer + (offset * 4 + 0) * sizeof(float)))) = b.w;
                (*((float *)(send_buffer + (offset * 4 + 1) * sizeof(float)))) = b.h;
                (*((float *)(send_buffer + (offset * 4 + 2) * sizeof(float)))) = b.x;
                (*((float *)(send_buffer + (offset * 4 + 3) * sizeof(float)))) = b.y;
                offset += 1;
                cout << b.w << " " << b.h << " " << b.x << " " << b.y << endl;
            }
            send(socket, send_buffer, send_buffer_size, 0);
            delete[] send_buffer;
            cout << "send data" << endl;
            shutdown(socket, SHUT_WR);
            close(socket);
            cout << "close socket" << endl;
            free_detections(dets, nboxes);
            free_image(im);
        }
    } catch (int error) {
        close(server_fd);
    }
}

void handler(int s) {
    close(server_fd);
    cout << "receve signup" << endl;
    exit(0);
}

int main(int argc, char **argv)
{
    if(argc != 4) {
        cout << "usage: " << argv[0] << " <cfg> <weights> <port>" << endl;
        exit(1);
    }
    signal(SIGKILL, handler);
    signal(SIGINT, handler);
    signal(SIGHUP, handler);
    detect_video(argv[1], argv[2], atoi(argv[3]));
    return 0;
}
