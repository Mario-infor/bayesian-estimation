#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <chrono>
#include <iostream>

#define drawCross( center, color, d )                                 \
line( img, Point( center.x - d, center.y - d ), Point( center.x + d, center.y + d ), color, 2, LINE_AA, 0); \
line( img, Point( center.x + d, center.y - d ), Point( center.x - d, center.y + d ), color, 2, LINE_AA, 0 )

using namespace cv;
using namespace std;


void on_mouseEvent(int event, int x, int y, int flags, void* p)
{
	vector < Point >* mD = (vector < Point > *)p;

	switch (event)
	{
	case EVENT_MOUSEMOVE:
		(*mD).insert((*mD).begin(), Point(x, y));
		break;
	}
}

void mouse()
{
	KalmanFilter KF(4, 2, 0);
	Point mousePos;
	Mat img(600, 800, CV_8UC3);
	vector < Point > mousev, kalmanv;
	double dtm, dts; //delta time in milliseconds and seconds.
	unsigned int n;
	char val;

	dtm = 10;
	dts = dtm * 10e-3;

	KF.transitionMatrix =
		(Mat_ < float >(4, 4) <<
			1, 0, dts, 0,
			0, 1, 0, dts,
			0, 0, 1, 0,
			0, 0, 0, 1);

	Mat_ < float >measurement(2, 1);

	measurement.setTo(Scalar(0));

	// Initialize the state
	KF.statePre.at < float >(0) = mousePos.x;
	KF.statePre.at < float >(1) = mousePos.y;
	KF.statePre.at < float >(2) = 0;
	KF.statePre.at < float >(3) = 0;

	// Initialize the noise matrix
	setIdentity(KF.measurementMatrix);
	KF.measurementMatrix *= 1;

	//Inicializamos las matrices de ruido
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));

	setIdentity(KF.measurementNoiseCov, Scalar::all(10));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// Image to show mouse tracking
	mousev.clear();
	kalmanv.clear();
	namedWindow("mouse kalman", 1);
	imshow("mouse kalman", img);
	waitKey(1);
	setMouseCallback("mouse kalman", on_mouseEvent, (void*)&mousev);

	n = 0;
	while (1)
	{
		if (mousev.size() > 1)
		{
			// First predict, to update the internal statePre variable
			KF.statePre = KF.transitionMatrix * KF.statePost;
			KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;

			Point predictPt(KF.statePre.at < float >(0), KF.statePre.at < float >(1));

			// Get mouse point
			measurement(0) = mousev[0].x;
			measurement(1) = mousev[0].y;
		}

		// Update the state from the last measurement.
		Mat temp = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
		Mat inverse;
		invert(temp, inverse, cv::DECOMP_LU);
		KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * inverse;

		KF.statePost = KF.statePre + KF.gain * (measurement - KF.measurementMatrix * KF.statePre);
		KF.errorCovPost = KF.errorCovPre - KF.gain * KF.measurementMatrix * KF.errorCovPre;

		// Plot the mouse trajectory in cyan, and the estimated trajectory in red.
		Point statePt(KF.statePost.at < float >(0), KF.statePost.at < float >(1));
		Point measPt(measurement(0), measurement(1));

		// Plot points
		imshow("mouse kalman", img);
		img = Scalar::all(0);

		mousev.push_back(measPt);
		kalmanv.push_back(statePt);
		drawCross(statePt, Scalar(255, 155, 0), 5);
		drawCross(measPt, Scalar(0, 0, 255), 5);

		for (uint i = 0; i < mousev.size() - 1; i++)
			line(img, mousev[i], mousev[i + 1], Scalar(255, 255, 0), 1);

		for (uint i = 0; i < kalmanv.size() - 1; i++)
			line(img, kalmanv[i], kalmanv[i + 1], Scalar(0, 155, 255), 1);

		val = waitKey((int)dtm);
		if (val == 27)
			break;
		n++;
	}
}

void mouse2() 
{
	KalmanFilter KF(6, 2, 0);
	Point mousePos;
	Mat img(600, 800, CV_8UC3);
	vector < Point > mousev, kalmanv;
	double dtm, dts, dts2; //delta time in milliseconda and seconds.
	unsigned int n;
	char val;

	dtm = 13.483;
	dts = dtm * 1e-3;
	dts2 = dts * dts;

	KF.transitionMatrix =
		(Mat_ < float >(6, 6) << 1, 0, dts, 0, 0.5 * dts2, 0, \
			0, 1, 0, dts, 0, 0.5 * dts2, \
			0, 0, 1, 0, dts, 0, \
			0, 0, 0, 1, 0, dts, \
			0, 0, 0, 0, 1, 0, \
			0, 0, 0, 0, 0, 1);

	Mat_ < float >measurement(2, 1);

	measurement.setTo(Scalar(0));

	KF.statePre.at < float >(0) = mousePos.x;
	KF.statePre.at < float >(1) = mousePos.y;
	KF.statePre.at < float >(2) = 0;
	KF.statePre.at < float >(3) = 0;
	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(10));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// Image to show mouse tracking
	mousev.clear();
	kalmanv.clear();
	namedWindow("mouse kalman", 1);
	imshow("mouse kalman", img);
	waitKey(1);
	setMouseCallback("mouse kalman", on_mouseEvent, (void*)&mousev);

	n = 0;
	auto start = chrono::steady_clock::now();

	Mat estimated;
	while (1)
	{
		if (mousev.size() > 1)
		{
			// First predict, to update the internal statePre variable
			KF.statePre = KF.transitionMatrix * KF.statePost;
			KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;

			Point predictPt(KF.statePre.at < float >(0),
				KF.statePre.at < float >(1));

			//Calculate time passed in milliseconds
			auto end = chrono::steady_clock::now();
			auto elapsed_milliseconds = chrono::duration_cast<std::chrono::milliseconds>(end - start);
			start = end;

			// Get mouse point
			measurement(0) = mousev[0].x;
			measurement(1) = mousev[0].y;

			if (n % 1 == 0)
			{
				// The update phase 
				Mat temp = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
				Mat inverse;
				invert(temp, inverse, cv::DECOMP_LU);
				KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * inverse;

				KF.statePost = KF.statePre + KF.gain * (measurement - KF.measurementMatrix * KF.statePre);
				KF.errorCovPost = KF.errorCovPre - KF.gain * KF.measurementMatrix * KF.errorCovPre;
			}

			if (n >= 1)
			{
				Point statePt(KF.statePost.at < float >(0), KF.statePost.at < float >(1));
				Point measPt(measurement(0), measurement(1));
				
				// Plot points
				imshow("mouse kalman", img);
				img = Scalar::all(0);

				mousev.push_back(measPt);
				kalmanv.push_back(statePt);
				drawCross(statePt, Scalar(255, 255, 255), 5);
				drawCross(measPt, Scalar(0, 0, 255), 5);

				for (uint i = 0; i < mousev.size() - 1; i++)
					line(img, mousev[i], mousev[i + 1], Scalar(255, 255, 0), 1);

				for (uint i = 0; i < kalmanv.size() - 1; i++)
					line(img, kalmanv[i], kalmanv[i + 1], Scalar(0, 155, 255), 1);
			}
		}
		val = waitKey((int)dtm);
		if (val == 27)
			break;
		n++;
	}
}


int main()
{
	// Choose between one example or the other
	mouse();
	
	//mouse2();

	return 0;
}
