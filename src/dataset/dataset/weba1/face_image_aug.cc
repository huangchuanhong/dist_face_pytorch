#include <stdlib.h>
#include <stdio.h>
#include "./global.hpp"
#include <assert.h>

struct FaceImageAugmentParam {
  int label_width;
  float mean_r;
  float mean_g;
  float mean_b;
  float scale;
  int height;
  int width;
  int channel;
  int batch_size;
  int resize_height;
  int resize_width;
  int patch_size;
  int patch_idx;
  bool do_aug;
  int FacePatchSize_Main;
  int FacePatchSize_Other;
  int PatchFullSize;
  int PatchCropSize;
  float illum_trans_prob;
  float gauss_blur_prob;
  float motion_blur_prob;
  float jpeg_comp_prob;
  float res_change_prob;
  float hsv_adjust_prob;};

extern "C" 
{
  unsigned char* FaceImageAugment(const FaceImageAugmentParam *param_p, char* img_file); 
}

cv::Rect GetFacePartRect(const cv::Size &imgSize, FACE_PART facePart);
bool ImageAugmentation(cv::Mat &img, AUG_METHOD augMethod, bool do_aug, const FaceImageAugmentParam & param_);
cv::Mat LoadPatches(cv::Mat & img, bool do_aug, const FaceImageAugmentParam & param_);
void ImgIllumTrans(cv::Mat &img, uint8_t nNewMean);
bool CompatibleMat(const cv::Mat &m1, const cv::Mat &m2);
cv::Mat GetMotionKernel(int sz, int d, float theta);
void ImgJpegComp(cv::Mat &img, int quality);
void ImgResChange(cv::Mat &img, float ratio);
void ImgHsvAdjust(cv::Mat &img, const int h, const int s, const int v);



unsigned char* FaceImageAugment(const FaceImageAugmentParam *param_p, char* img_file){
    //do resize based on src
    const FaceImageAugmentParam &param_ = *param_p;
    cv::Mat src = cv::imread(img_file);
    cv::Mat res;
    //cv::resize(src, res, cv::Size(param_.resize_width, param_.resize_height));
    if (param_.resize_height != -1 && param_.resize_width != -1) {
      int new_width = param_.resize_height;
      int new_height = param_.resize_width;
      if (new_width != src.cols || new_height != src.rows) {
            std::cout << "doing resize: old size " << src.cols << " " << src.rows << std::endl;
        cv::resize(src, res, cv::Size(new_width, new_height));
      } else {
        res = src.clone();
      }
    } else {
      res = src.clone();
    }
    bool do_aug = param_.do_aug;
    cv::Mat aug_img = LoadPatches(res, do_aug, param_);
    return aug_img.ptr<uchar>(0);
} 

cv::Rect GetFacePartRect(
       const cv::Size &imgSize,
       FACE_PART facePart
       )
  {
      cv::Size_<float> m_FaceFullSize = cv::Size_<float>(267, 267);
      cv::Size_<float> m_FaceRimlessSize = cv::Size_<float>(200, 200);
      cv::Size_<float> m_FacePartSize = cv::Size_<float>(128, 128);
      //int m_FaceRimlessSize = 300;
      //int m_PatchCropSize = 192;

      switch (facePart)
      {
      case FP_FULL_FACE:
          return RectOfCenter(CenterOfSize(cv::Size2f(imgSize)), m_FaceFullSize);
      }
  }

bool ImageAugmentation(
    cv::Mat &img,
    AUG_METHOD augMethod,
    bool do_aug, 
    const FaceImageAugmentParam & param_
    )
{
    assert(img.type() == CV_8UC3);
    assert(augMethod < AM_GUARD);
    auto UNIPROBTEST = [](float fProb) -> bool
    {
        return (std::rand() / (RAND_MAX + 1.0f)) < fProb;
    };

    float illum_trans_prob = param_.illum_trans_prob;
    float gauss_blur_prob = param_.gauss_blur_prob;
    float motion_blur_prob = param_.motion_blur_prob;
    float jpeg_comp_prob = param_.jpeg_comp_prob;
    float res_change_prob = param_.res_change_prob;
    float hsv_adjust_prob = param_.hsv_adjust_prob;
    if (!do_aug) {
        illum_trans_prob = 0;
        gauss_blur_prob = 0;
        motion_blur_prob = 0;
        jpeg_comp_prob = 0;
        res_change_prob = 0;
        hsv_adjust_prob = 0;
    }
    if (UNIPROBTEST(illum_trans_prob))
    {
        //CMyTimer t;
        std::uniform_int_distribution<> rMean(50, 200);
        int nNewMean = rMean(m_rg);
        ImgIllumTrans(img, nNewMean);
    }
    if (UNIPROBTEST(gauss_blur_prob))
    {
        //CMyTimer t;
        std::uniform_int_distribution<> rSize(1, 2);
        int nKernelSize = rSize(m_rg) * 2 + 1; // must be 3 or 5
        cv::GaussianBlur(img, img, cv::Size(nKernelSize, nKernelSize), 0, 0);
    }
    if (UNIPROBTEST(motion_blur_prob))
    {
        // NEEDS REVIEW!
        const int sz = 10;
        std::uniform_int_distribution<> rSize(0, sz - 3);
        std::uniform_real_distribution<float> rTheta(0.0f, 180.0f);
        cv::Mat kernel = GetMotionKernel(sz, rSize(m_rg) + 1, rTheta(m_rg));
        cv::filter2D(img, img, img.depth(), kernel);
    }
    if (UNIPROBTEST(jpeg_comp_prob))
    {
        //CMyTimer t;
        std::uniform_int_distribution<> rQuality(30, 60);
        ImgJpegComp(img, rQuality(m_rg));
        //std::cout << "jpeg " << t.Reset() << std::endl;
    }
    if (UNIPROBTEST(res_change_prob))
    {
        //CMyTimer t;
        std::uniform_real_distribution<float> rScale(0.2f, 1.0f);
        ImgResChange(img, rScale(m_rg));
        //std::cout << "res " << t.Reset() << std::endl;
    }
    if (UNIPROBTEST(hsv_adjust_prob))
    {
        //CMyTimer t;
        std::uniform_int_distribution<> rh(-10, 10);
        std::uniform_int_distribution<> rs(-20, 20);
        ImgHsvAdjust(img, rh(m_rg), rs(m_rg), 0);
    }
    return true;
}


cv::Mat LoadPatches(
    cv::Mat & img,
    bool do_aug,
    const FaceImageAugmentParam & param_
    )
{
    //Do data augmentation
    std::uniform_int_distribution<> rAugM(0, AM_GUARD - 1);
    AUG_METHOD augMethod = (AUG_METHOD)rAugM(m_rg);
    if (!ImageAugmentation(img, augMethod, do_aug, param_))
    {
        augMethod = (AUG_METHOD)-1;
    }

    int szw = param_.PatchFullSize;
    int szh = param_.PatchFullSize;
    cv::Size m_PatchFullSize = cv::Size(szw, szh);
    int imgw = param_.PatchCropSize;
    int imgh = param_.PatchCropSize;
    cv::Size m_PatchCropSize = cv::Size(imgw, imgh);
    bool patch_rand_crop = true;
    if (!do_aug) patch_rand_crop = false;
   
    cv::Rect partRect(GetFacePartRect(img.size(), FP_FULL_FACE)); 
    cv::Rect srcRect(cv::Rect(cv::Point2f(), img.size()) & partRect);
    cv::Rect dstRect(srcRect.tl() - partRect.tl(), srcRect.size());
    cv::Mat part = cv::Mat::zeros(partRect.size(), img.type());
    img(srcRect).copyTo(part(dstRect));
    cv::resize(part, part, m_PatchFullSize);
    cv::Size border = m_PatchFullSize - m_PatchCropSize;
    cv::Point cropTL(border / 2);
    if (patch_rand_crop)
    {
        cropTL = cv::Point(rand() % border.width, rand() % border.height);
    }
    cv::Mat aug_img = part(cv::Rect(cropTL, m_PatchCropSize)).clone();
    return aug_img;
}
 

void ImgIllumTrans(cv::Mat &img, uint8_t nNewMean)
{
        if (img.channels() == 3) {
                cv::cvtColor(img, img, CV_BGR2HLS);
                std::vector<cv::Mat> hls(3);
                cv::split(img, hls);
                double dMinL, dMaxL;
                cv::minMaxLoc(hls[1], &dMinL, &dMaxL);
                int nNewRange = 255;
                int nNewMinL = nNewMean - nNewRange / 2;
                double alpha = (double)nNewRange / (double)(dMaxL - dMinL);
                hls[1].convertTo(
                        hls[1],
                        hls[1].type(),
                        alpha,
                        nNewMinL - dMinL * alpha
                        );
                cv::merge(hls, img);
                cv::cvtColor(img, img, CV_HLS2BGR);
        } else {
                double dMinL, dMaxL;
                cv::minMaxLoc(img, &dMinL, &dMaxL);
                int nNewRange = 255;
                int nNewMinL = nNewMean - nNewRange / 2;
                double alpha = (double)nNewRange / (double)(dMaxL - dMinL);
                // double beta = (double)nNewMinL - dMinL * alpha;
                img.convertTo(
                        img,
                        img.type(),
                        alpha,
                        nNewMinL - dMinL * alpha
                        );
                // for (int row = 0; row < img.rows; ++row) {
                //         for (int col = 0; col < img.cols; ++col) {
                //                 float gray = img.at<uchar>(row, col);
                //                 img.at<uchar>(row, col) = cv::saturate_cast<uchar>(gray * alpha + beta);
                //         }
                // }
        }

}

bool CompatibleMat(const cv::Mat &m1, const cv::Mat &m2)
{
        return (
                        m1.size() == m2.size() &&
                        m1.channels() == m2.channels() &&
                        m1.type() == m2.type()
                        );
}

cv::Mat GetMotionKernel(int sz, int d, float theta)
{
        const float c = std::cos(theta / 180.0f * (float)M_PI);
        const float s = std::sin(theta / 180.0f * (float)M_PI);

        cv::Mat A = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
        A.at<float>(0, 0) = c;
        A.at<float>(0, 1) = -s;
        A.at<float>(1, 0) = s;
        A.at<float>(1, 1) = c;

        cv::Mat vec = cv::Mat::zeros(cv::Size(1, 2), CV_32FC1);
        vec.at<float>(0, 0) = (d - 1) / 2;

        cv::Mat B = A(cv::Rect(0, 0, 2, 2)).clone();
        cv::Mat tmp = B * vec;
        A.at<float>(0, 2) = sz / 2 - tmp.at<float>(0, 0);
        A.at<float>(1, 2) = sz / 2 - tmp.at<float>(1, 0);

        cv::Mat kernel;
        cv::warpAffine(
                        cv::Mat::ones(cv::Size(d, 1), CV_32FC1),
                        kernel,
                        A,
                        cv::Size(sz, sz),
                        cv::INTER_CUBIC
                        );

        kernel /= cv::sum(kernel)[0];
        return kernel;
}

void ImgJpegComp(cv::Mat &img, int quality)
{
        std::vector<int> params;
        params.push_back(CV_IMWRITE_JPEG_QUALITY);
        params.push_back(quality); //jpeg quality 0-100, the higher the better

        std::vector<uchar> buffer;
        if (cv::imencode(".jpg", img, buffer, params))
        {
                img = cv::imdecode(buffer, CV_LOAD_IMAGE_COLOR);
        }
}


void ImgResChange(cv::Mat &img, float ratio)
{
        assert(ratio <= 1 && ratio > 0);
        cv::Size orgSize = img.size();
        cv::Size smallSize(
                        std::max(1, int(img.cols * ratio)),
                        std::max(1, int(img.rows * ratio))
                        );
        cv::resize(img, img, smallSize);        //down sampling
        cv::resize(img, img, orgSize);          //up sampling
}

void ImgHsvAdjust(cv::Mat &img, const int h, const int s, const int v)
{
        cv::Mat hsv;
        if (img.channels() == 1) return;
        if (img.channels() == 4)
        {
                cv::cvtColor(img, hsv, cv::COLOR_RGBA2BGR);
        }
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

        for (int i = 0; i < hsv.rows; i++)
        {
                cv::Vec3b *pRow = hsv.ptr<cv::Vec3b>(i);
                for (int j = 0; j < hsv.cols; j++)
                {
                        cv::Vec3b &pixel = pRow[j];
                        int cPointH = pixel[0] + h;
                        int cPointS = pixel[1] + s;
                        int cPointV = pixel[2] + v;
                        // hue
                        if (cPointH < 0)
                        {
                                pixel[0] = 0;
                        }
                        else if (cPointH > 179)
                        {
                                pixel[0] = 179;
                        }
                        else
                        {
                                pixel[0] = cPointH;
                        }
                        // saturation
                        if (cPointS < 0)
                        {
                                pixel[1] = 0;
                        }
                        else if (cPointS > 255)
                        {
                                pixel[1] = 255;
                        }
                        else
                        {
                                pixel[1] = cPointS;
                        }
                        // value
                        if (cPointV < 0)
                        {
                                pixel[2] = 0;
                        }
                        else if (cPointV > 255)
                        {

                                pixel[2] = 255;
                        }
                        else
                        {
                                pixel[2] = cPointV;
                        }
                }
        }
        cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
}

