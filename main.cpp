/**
    prida
    main.cpp
    @author Tianyi Shan
    @version 1.0 May/17/2018
*/
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;
using namespace std;

int channel;
double LAMBDA;
int KERNEL_SIZE;
char* PATH;

struct ctf_params_t{
    double lambdaMultiplier;
    double maxLambda;
    double finalLambda, kernelSizeMultiplier;
};

struct uk_t{
    cv::Mat u;
    cv::Mat k;
};

struct params_t{
    double MK;
    double NK;
    double niters;
};

struct input {
    cv::Mat f;
    double MK ;
    double NK ;
    double lambda ;
    double lambdaMultiplier ;
    double scaleMultiplier ;
    double largestLambda ;
};

struct output{
    vector<cv::Mat> fp;
    vector<double> Mp, Np, MKp, NKp, lambdas;
    int  scales;
} ;

enum ConvolutionType {
            CONVOLUTION_FULL,
            CONVOLUTION_VALID
};

/**********************************************************************************************
 * @param img The input img, Kernel The input kernel, type FULL or VALID, dest The output result
 * Compute conv2 by calling filter2D.
 ************************************************************************************************/
void conv2(  const cv::Mat &img,   const cv::Mat& kernel, ConvolutionType type, cv::Mat& dest) {
    Mat flipped_kernel;
    flip( kernel, flipped_kernel, -1 );

    Point2i pad;
    Mat padded, padded2;

    switch( type ) {
        case CONVOLUTION_VALID:
            padded = img;
            pad = Point2i( kernel.cols - 1, kernel.rows - 1);
            break;

        case CONVOLUTION_FULL:
            pad = Point2i( kernel.cols - 1, kernel.rows - 1);
            padded.create(img.rows + 2*(kernel.rows - 1), img.cols + 2*(kernel.cols - 1), img.type());
            padded.setTo(cv::Scalar::all(0));
            img.copyTo(padded(Rect((kernel.rows - 1), (kernel.cols - 1), img.cols, img.rows)));
            break;

        default:
            throw runtime_error("Unsupported convolutional shape");
    }
    Rect region = Rect( pad.x / 2, pad.y / 2, padded.cols - pad.x, padded.rows - pad.y);
    filter2D( padded, dest , -1, flipped_kernel, Point(-1, -1), 0, BORDER_CONSTANT );

    dest = dest( region );
}

/***********************************************************************************
 * @param f The scaled input image. dest The result image.
 * Calc total variation for image f
 *
 **********************************************************************************/
void gradTVcc(const cv::Mat &f, cv::Mat &dest){
    cv::Mat fxforw(f.size(), f.type());
    f(cv::Range(1,f.rows),cv::Range(0,f.cols)).copyTo(fxforw);
    cv::copyMakeBorder(fxforw, fxforw,0, 1, 0, 0, cv::BORDER_REPLICATE  );
    fxforw = fxforw - f;

    cv::Mat  fyforw(f.size(), f.type());
    f(cv::Range(0,f.rows),cv::Range(1,f.cols)).copyTo(fyforw);
    cv::copyMakeBorder(fyforw, fyforw,0, 0, 0, 1, cv::BORDER_REPLICATE  );
    fyforw = fyforw - f;

    cv::Mat fxback(f.size(), f.type());
    f(cv::Range(0,f.rows-1),cv::Range(0,f.cols)).copyTo(fxback);
    cv::copyMakeBorder(fxback, fxback,1, 0, 0, 0, cv::BORDER_REPLICATE  );

    cv::Mat fyback(f.size(), f.type());
    f(cv::Range(0,f.rows),cv::Range(0,f.cols-1)).copyTo(fyback);
    cv::copyMakeBorder(fyback, fyback,0, 0, 1, 0, cv::BORDER_REPLICATE  );

    cv::Mat fxmixd(f.size(), f.type());
    f(cv::Range(1,f.rows),cv::Range(0,f.cols-1)).copyTo(fxmixd);
    cv::copyMakeBorder(fxmixd, fxmixd,0, 1, 1, 0, cv::BORDER_REPLICATE  );
    fxmixd = fxmixd - fyback;

    cv::Mat fymixd(f.size(), f.type());
    f(cv::Range(0,f.rows-1),cv::Range(1,f.cols)).copyTo(fymixd);
    cv::copyMakeBorder(fymixd, fymixd,1, 0, 0, 1, cv::BORDER_REPLICATE  );
    fymixd = fymixd - fxback;
    fyback = f - fyback ;
    fxback = f - fxback;

    dest = cv::Mat::zeros(f.size(), CV_64FC3);
    vector<cv::Mat> pdest;
    vector<cv::Mat> pfxforw, pfyforw, pfxback, pfyback, pfxmixd, pfymixd;
    cv::split(fxforw, pfxforw);
    cv::split(fyforw, pfyforw);
    cv::split(fxback, pfxback);
    cv::split(fyback, pfyback);
    cv::split(fxmixd, pfxmixd);
    cv::split(fymixd, pfymixd);
    cv::split(dest, pdest);

    int c = 0;
    while (c < channel) {

        cv:: Mat powfx;
        cv::pow(pfxforw[c],2,powfx);
        cv:: Mat powfy;
        cv::pow(pfyforw[c],2,powfy);
        cv:: Mat sqtforw;

        cv::sqrt(powfx + powfy  ,sqtforw);

        cv:: Mat powfxback;
        cv::pow(pfxback[c],2,powfxback);
        cv:: Mat powfymixd;
        cv::pow(pfymixd[c],2,powfymixd);

        cv:: Mat sqtmixed;
        cv::sqrt(powfymixd  + powfxback  ,sqtmixed);

        cv:: Mat powfxmixd;
        cv::pow(pfxmixd[c],2,powfxmixd);
        cv:: Mat powfyback;
        cv::pow(pfyback[c],2,powfyback);
        cv:: Mat sqtback;

        cv::sqrt(powfxmixd  + powfyback,sqtback);

        cv:: Mat max1;
        cv::max( sqtforw,1e-3, max1);
        cv:: Mat max2;
        cv::max( sqtmixed,1e-3, max2);
        cv:: Mat max3;
        cv::max( sqtback,1e-3, max3);

        pdest[c] = (pfxforw[c] + pfyforw[c]) / max1;
        pdest[c] = pdest[c] - pfxback[c]  /  max2;
        pdest[c] = pdest[c] - pfyback[c] / max3;

        c++;
    }
    if (c == 1){
        Mat t[] = {pdest[0], pdest[0], pdest[0]};
        cv::merge(t, 3, dest);
    }else {
        cv::merge(pdest, dest);
    }
}

/***********************************************************************************
 * @param f The scaled input image. u The scaled result image.
 *        k The scaled result kernel. lambda The input lambda. params Parameters.
 *        uk The output image and kernel.
 * Initialize gradu and gradk
 * Loop for niters times, call conv2 and gradTVCC to get the result of u and k.
 **********************************************************************************/
void prida(cv::Mat &f, cv::Mat &u, cv::Mat &k, const double lambda, struct params_t params ) {
    cv::Mat gradu = cv::Mat::zeros(cv::Size(f.cols + (int) params.NK - 1,
                                            f.rows + (int) params.MK - 1), CV_64FC3);
    cv::Mat gradk = cv::Mat::zeros(k.size(), CV_64F);

    for (int i = 0; i < params.niters; i++) {
        gradu = cv::Mat::zeros(cv::Size(f.cols + (int) params.NK - 1,
                                        f.rows + (int) params.MK - 1), CV_64FC3);
        int c = 0;
        if (channel == 1){
            vector<cv::Mat> pGradu, pf, pu;
            cv::split(gradu, pGradu);
            cv::split(f, pf);
            cv::split(u, pu);

            cv:: Mat tmp;
            conv2(pu[c], k, CONVOLUTION_VALID,tmp);
            tmp = tmp - pf[c];

            cv::Mat rotk = cv::Mat::zeros(k.size(), CV_64F);
            cv::rotate(k,rotk, cv::ROTATE_180);

            conv2(tmp , rotk, CONVOLUTION_FULL,pGradu[c]);
            Mat t[] = {pGradu[c], pGradu[c], pGradu[c]};
            cv::merge(t, 3, gradu);
        }else if(channel == 3){
            vector<cv::Mat> pGradu, pf, pu;
            cv::split(gradu, pGradu);
            cv::split(f, pf);
            cv::split(u, pu);
            while (c < 3) {
                cv:: Mat tmp;
                conv2(pu[c], k, CONVOLUTION_VALID,tmp);
                tmp = tmp - pf[c];
                cv::Mat rotk = cv::Mat::zeros(k.size(), CV_64F);
                cv::rotate(k,rotk, cv::ROTATE_180);
                conv2(tmp , rotk, CONVOLUTION_FULL,pGradu[c]);
                c++;
            }
            cv::merge(pGradu, gradu);
        }
        c = 0;

        cv::Mat gradTV = cv::Mat::zeros(u.size(), CV_64F);
        gradTVcc(u, gradTV);
        gradu = (gradu - lambda*gradTV);

        double minValu;
        double maxValu;
        cv::minMaxLoc(u, &minValu, &maxValu);

        double minValgu;
        double maxValgu;
        cv::minMaxLoc(cv::abs(gradu), &minValgu, &maxValgu);

        double sf = 1e-3 * maxValu / max(1e-31, maxValgu);
        cv::Mat u_new;
        u_new   = u - sf*gradu;
        gradk = cv::Mat::zeros(gradk.size(), CV_64FC1);

        vector<cv::Mat>  pff, puu;
        cv::split(f, pff);
        cv::split(u, puu);

        while (c < channel) {
            cv::Mat subconv2;
            conv2(puu[c], k, CONVOLUTION_VALID, subconv2);
            subconv2 = subconv2 - pff[c];

            cv::Mat rotu;
            cv::rotate(puu[c],rotu, cv::ROTATE_180);

            cv::Mat majconv2;
            conv2(rotu, subconv2, CONVOLUTION_VALID, majconv2);
            gradk = gradk + majconv2;
            c++;
        }
        double minValk,  maxValk;
        cv::minMaxLoc(k, &minValk, &maxValk);
        double minValgk, maxValgk;
        cv::minMaxLoc(cv::abs(gradk), &minValgk, &maxValgk);

        double sh = 1e-3 * maxValk / max(1e-31, maxValgk);
        double eps = DBL_EPSILON;
        cv::Mat etai = sh / (k + eps);

        int bigM = 1000;
        cv::Mat expTmp;
        cv::exp((-etai).mul(gradk), expTmp);

        cv::Mat tmp2 = cv::min(expTmp, bigM);
        cv::Mat MDS;
        MDS = k.mul(tmp2);

        cv::Mat k_new = MDS/cv::sum(MDS)[0];

        u = u_new;
        k = k_new;
    }
}

/*****************************************************************************************************************
 * @param data Including input image, kernel size, lambda, lambdaMultiplier, scaleMultiplier and largestLambda
 *        answer The dest struct stores all the result
 ****************************************************************************************************************/
struct output buildPyramid(struct input &data, struct output &answer){

    double smallestScale = 3;
    int scales = 1;
    double mkpnext = data.MK;
    double nkpnext = data.NK;
    double lamnext = data.lambda;


    cv::Size s = data.f.size();
    int M = s.height;
    int N = s.width;

    while (mkpnext > smallestScale && nkpnext > smallestScale && lamnext * data.lambdaMultiplier
                                                                 < data.largestLambda) {
        scales = scales + 1;
        double lamprev = lamnext;
        double mkpprev = mkpnext;
        double nkpprev = nkpnext;

        // Compute lambda value for the current scale
        lamnext = lamprev * data.lambdaMultiplier;
        mkpnext = round(mkpprev / data.scaleMultiplier);
        nkpnext = round(nkpprev / data.scaleMultiplier);

        // Makes kernel dimension odd
        if (fmod(mkpnext, 2) == 0)
            mkpnext = mkpnext - 1;

        if (fmod(nkpnext,2) == 0)
            nkpnext = nkpnext - 1;

        if (nkpnext == nkpprev)
            nkpnext = nkpnext - 2;

        if (mkpnext == mkpprev)
            mkpnext = mkpnext - 2;

        if (nkpnext < smallestScale)
            nkpnext = smallestScale;

        if (mkpnext < smallestScale)
            mkpnext = smallestScale;
    }

    answer.fp.resize(scales);
    answer.Mp.resize(scales);
    answer.Np.resize(scales);
    answer.MKp.resize(scales);
    answer.NKp.resize(scales);
    answer.lambdas.resize(scales);

//set the first (finest level) of pyramid to original data
    answer.fp[0] = data.f;
    answer.Mp[0] = M;
    answer.Np[0] = N;
    answer.MKp[0] = data.MK;
    answer.NKp[0] = data.NK;
    answer.lambdas[0] = data.lambda;

    //loop and fill the rest of the pyramid
    for (int s = 1 ; s <scales; s++){
        answer.lambdas[s] = answer.lambdas[s - 1] * data.lambdaMultiplier;

        answer.MKp[s] = round(answer.MKp[s - 1] / data.scaleMultiplier);
        answer.NKp[s] = round(answer.NKp[s - 1] / data.scaleMultiplier);

        // Makes kernel dimension odd
        if (fmod(answer.MKp[s],2) == 0)
            answer.MKp[s] = answer.MKp[s] - 1;

        if (fmod(answer.NKp[s],2) == 0)
            answer.NKp[s] -= 1;

        if (answer.NKp[s] == answer.NKp[s-1])
            answer.NKp[s] -= 2;

        if (answer.MKp[s] == answer.MKp[s-1])
            answer.MKp[s] -= 2;

        if (answer.NKp[s] < smallestScale)
            answer.NKp[s] = smallestScale;

        if (answer.MKp[s] < smallestScale)
            answer.MKp[s] = smallestScale;

        //Correct scaleFactor for kernel dimension correction
        double factorM = answer.MKp[s-1]/answer.MKp[s];
        double factorN = answer.NKp[s-1]/answer.NKp[s];

        answer.Mp[s] = round(answer.Mp[s-1] / factorM);
        answer.Np[s] = round(answer.Np[s-1] / factorN);

        // Makes image dimension odd
        if (fmod(answer.Mp[s],2) == 0)
            answer.Mp[s] -= 1;

        if (fmod(answer.Np[s],2) == 0)
            answer.Np[s] -= 1;

        cv:: Mat dst ;
        cv::resize(data.f, dst, cv::Size((int) (answer.Np[s]), (int)(answer.Mp[s])) , 0, 0, cv::INTER_LINEAR);
        answer.fp[s] = dst;
    }
    answer.scales = scales;
    return answer;
}

/***********************************************************************************************
 * @param f The input image. blind_params The kernel size and niters size. params Parameters
 *        uk The output image and kernel
 * Call buildPyrmaid
 * For each layer in the pyrmaid, call Prida
 **********************************************************************************************/
void coarseToFine(cv::Mat f, struct params_t blind_params, struct ctf_params_t params, struct uk_t &uk ){
    double MK = blind_params.MK;
    double NK = blind_params.NK;

    cv:: Mat u;
    int top = (int)floor(MK/2);
    int left = (int )floor(NK/2);
    cv::copyMakeBorder(f, u,top, top, left, left, cv::BORDER_REPLICATE  );

    cv::Mat k = cv::Mat::ones(cv::Size((int)MK,(int)NK),CV_64F);
    k = k/MK/NK;

    struct input data;
    struct output answer;
    data.MK = MK;
    data.NK = NK;
    data.lambda = params.finalLambda;
    data.lambdaMultiplier = params.lambdaMultiplier;
    data.scaleMultiplier = params.kernelSizeMultiplier;
    data.largestLambda = params.maxLambda;
    data.f = f;
    buildPyramid(data , answer);

    for (int i = answer.scales-1; i >=0; i--){
        double Ms, Ns, MKs, NKs, lambda;
        cv::Mat fs;
        Ms = answer.Mp[i];
        Ns = answer.Np[i];

        MKs = answer.MKp[i];
        NKs = answer.NKp[i];
        fs = answer.fp[i];

        lambda = answer.lambdas[i];

        cv::resize(u, u, cv::Size((int) (Ns + NKs - 1), (int)(Ms + MKs - 1)) , 0, 0, cv::INTER_LINEAR);
        cv::resize(k, k, cv::Size((int) NKs, (int)MKs) , 0, 0, cv::INTER_LINEAR);

        k = k/cv::sum(k)[0];
        blind_params.MK = MKs;
        blind_params.NK = NKs;

        prida(fs, u, k, lambda, blind_params);
        cout<< "Working on Scale: " << i+1 <<  " with lambda = "<< data.lambda <<" with pyramid_lambda = " << lambda << " and Kernel size " << MKs << endl;
    }
    uk.u = u;
    uk.k = k;
}

/********************************************************************************************
 * @param f The input image. lambda The input lambda. params The kernel size and niter size
 *        uk The output image and kernel
 * Convert the image to double precision
 * Adjust the photo size
 * Initialize parameters
 * Call coarseToFine
 *********************************************************************************************/
void blind_deconv(cv::Mat &f, double &lambda,struct params_t &params, struct uk_t &uk ){
    f.convertTo( f, CV_64F, 1./255. );
    cv:: Mat f3;
    int rpad = 0;
    int cpad = 0;
    if(fmod(f.rows,2) == 0){
        rpad = 1;
    }
    if(fmod(f.cols,2) == 0){
        cpad = 1;
    }
    f(cv::Range(0,f.rows-rpad),cv::Range(0,f.cols-cpad) ).copyTo(f3);
    struct ctf_params_t ctf_params;
    ctf_params.lambdaMultiplier = 1.9;
    ctf_params.maxLambda = 1.1e-1;
    ctf_params.finalLambda = lambda;
    ctf_params.kernelSizeMultiplier = 1.1;
    coarseToFine(f3, params, ctf_params, uk);
}

/***********************************************
 * @param img The input image
 *Check the color channel.
 *
 ***********************************************/
bool isGrayImage( Mat img ){
    Mat dst;
    Mat bgr[3];
    split( img, bgr );
    absdiff( bgr[0], bgr[1], dst );
    if(countNonZero( dst ))
        return false;
    absdiff( bgr[0], bgr[2], dst );
    return !countNonZero( dst );
}

/***********************************************
 * @param image The input image
 * Create structs uk and params
 * Set the niter number to 1000
 * Call blind_deconv
 * Write out the results to the folder
 *
 ***********************************************/
void helper(cv::Mat image ){
    struct uk_t uk;
    struct params_t params;

    params.MK = KERNEL_SIZE; // row
    params.NK = KERNEL_SIZE; // col
    params.niters = 1000;

    blind_deconv(image, LAMBDA, params,uk);
    std::ostringstream name;

    PATH[strlen(PATH) - 4] = '\0';
    name << PATH << "recov.png";
    std::ostringstream kname;
    kname << PATH << "recovkernel.png";

    cv::Mat tmpk, tmpu;
    uk.u.convertTo(tmpu,CV_8U, 1.*255.);
    double ksml, klag;
    cv::minMaxLoc(uk.k,&ksml, &klag);

    tmpk = uk.k / klag;
    uk.k.convertTo(tmpk,CV_8U, 1.*255.);
    cv::applyColorMap( tmpk, tmpk, COLORMAP_BONE);
    cv::imwrite( name.str(), tmpu);
    cv::imwrite( kname.str(),  tmpk );
}


/***********************************************
 *  Load the Image
 *  Convert the param to double numbers
 *  Determine channel number
 *  Call threadhelp
 ***********************************************/
int main(int argc, char* argv[]) {
    if(argc < 4) {
        printf("usage: deblur IMAGE_PATH LAMBDA KERNEL_SIZE");
        exit(1);
    }
    LAMBDA = atof(argv[2]);
    KERNEL_SIZE = atof(argv[3]);
    PATH = argv[1];

    cv::Mat image = cv::imread(PATH);

    if(isGrayImage(image)){
        channel = 1;
        cout<<"The image is gray" <<endl;
    }else{
        channel = 3;
        cout <<"The image is color " <<endl;
    }
    helper(image);
    return 0;
}