#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include "Circle.h"
#include <cmath>
#include <Eigen/Dense>

#define SLIDE_MAX 1000					//!< El valor maximo de pasos del control de barra deslizante del GUI.

using namespace std;
using namespace cv;

#define IM_WIDTH 640						//!< El ancho de la imagen.
#define IM_HEIGHT 480						//!< Lo alto de la imagen.


#define  a_COMPONENT  30.677		//!< El valor del componente a del modelo de color.
#define  b_COMPONENT 58.212			//!< El valor del componente b del modelo de color.

/*!
\fn void printMat(Mat &M, const char *name = NULL, bool transp=false)
\brief Esta funcion imprime en stdout como texto el contenido de una matriz
	   almacenado en un objeto CV::Mat.

\param M La matriz a ser impresa
\param name El nombre de la variable o identificador asociado a la matriz.
\param transp Indica si se imprime la matriz de manera convencional o
	   transpuesta.

 Esta funcion imprime en stdout como texto el contenido de una matriz almacenado en un objeto CV::Mat.
 El formato de impresión sigue la sintaxis aceptada por Matlab/octave y tiene
 una opcion para imprimir en forma transpuesta (añadiendo un apostrofe al final
 de la definicion), lo cual es útil cuando se imprimen vectores columna en forma
 de renglones.

*/
template < typename X >
void printMat(Mat& M, const char* name = NULL, bool transp = false)
{
	int i, j;

	if (name)
		cout << name << " = [";
	else
		cout << name << "[";

	if (transp)
	{
		for (i = 0; i < M.cols; ++i)
		{
			for (j = 0; j < M.rows - 1; ++j)
				cout << M.at < X >(i, j) << ", ";
			cout << M.at < X >(i, j) << endl;
		}
		cout << "]'" << endl;
	}
	else
	{
		for (i = 0; i < M.rows; ++i)
		{
			for (j = 0; j < M.cols - 1; ++j)
				cout << M.at < X >(i, j) << ", ";
			cout << M.at < X >(i, j) << endl;
		}
		cout << "]" << endl;
	}
}

/*!
\fn int MeaniCov(Mat &image, Mat &Mask, Mat &mean, Mat &cov)

\brief Esta funcion calcula la media y la inversa de la matriz de covarianza de
cada uno de los elementos de una matriz que representa una imagen a color.
\param image La imagen a la que se le va a calcular la media y la matriz de
			 covarianza.
\param Mask Una matriz que binaria que se usa para determinar sobre que
			elementos de imagen se va a realizar el cómputo.
\param mean Una matriz de 3 renglones y 1 columna, en donde se regresa el vector
			promedio de los pixeles de la imagen.
\param mean Una matriz de 3 renglones y 3 columnas, en donde se regresa la
			matriz de covarianza de los pixeles de la imagen.
\return El número de elementos procesados. Regresa el valor -1 si el número de
		elementos a procesar es menor a 2.

Esta funcion calcula la media y la inversa de la matriz de covarianza de cada
uno de los elementos de una matriz que representa una imagen a color. Como
parámetro se recibe una matriz de mascara, del mismo tamaño que la imagen de
entrada, con lo cual podemos controlar de manera fina que elementos de la matriz
de entrada se deben procesar. La funcion regresa el número de elementos
procesados (que es el número de elementos de la matriz de mascara diferentes a
0); si ese número es menor a 1, la funcion regresa el valor -1 para indicar un
fallo, y como no se puede calcular ni el vector promedio ni la matriz de
covarianza los valores de los parámteros por referencia mean y cov son
indeterminados. Si el número de elementos procesado es igual a 1, el parametro
por referencia cov es valido y contiene el valor del elemento procesado, pero el
parametro por referencia cov es indeterminado, y la funcion indica fallo
regresando el valor -1.
*/

int MeaniCov(Mat& image, Mat& Mask, Mat& mean, Mat& icov)
{
	float m[2], pm[2], Cv[3], iCont, iFact;
	int cont;
	mean = Mat::zeros(2, 1, CV_32F);
	icov = Mat::zeros(2, 2, CV_32F);
	Mat_ < Vec3f >::iterator it, itEnd;
	Mat_ < uchar >::iterator itM;

	it = image.begin < Vec3f >();
	itM = Mask.begin < uchar >();
	itEnd = image.end < Vec3f >();
	m[0] = m[1] = 0;
	memset(m, 0, 2 * sizeof(float));
	for (cont = 0; it != itEnd; ++it, ++itM)
	{
		if ((*itM))
		{
			m[0] += (*it)[1];
			m[1] += (*it)[2];
			cont++;
		}
	}

	if (!cont)
		return -1;
	m[0] /= cont;
	m[1] /= cont;
	mean = Mat(2, 1, CV_32F, m).clone();

	if (cont < 1)
	{
		icov.at < float >(0, 0) = icov.at < float >(1, 1) =
			icov.at < float >(2, 2) = 1.;
		return -1;
	}
	it = image.begin < Vec3f >();
	itM = Mask.begin < uchar >();
	memset(Cv, 0, 3 * sizeof(float));
	for (; it != itEnd; ++it, ++itM)
	{
		if ((*itM))
		{
			pm[0] = (*it)[1] - m[0];
			pm[1] = (*it)[2] - m[1];
			Cv[0] += pm[0] * pm[0];
			Cv[1] += pm[1] * pm[1];
			Cv[2] += pm[0] * pm[1];
		}
	}
	cont--;
	iCont = 1. / cont;
	Cv[0] *= iCont;
	Cv[1] *= iCont;
	Cv[2] *= iCont;

	iFact = 1. / (Cv[0] * Cv[1] - Cv[2] * Cv[2]);
	icov.at < float >(0, 0) = Cv[1] * iFact;
	icov.at < float >(1, 1) = Cv[0] * iFact;
	icov.at < float >(1, 0) = icov.at < float >(0, 1) = Cv[2] * iFact;

	return cont;
}

/*!
\struct barData
\brief Esta estructura almacena el valor inicial y el factor de incremento de
una barra deslizante (un elemento del GUI usada en el programa).
*/
struct barData
{
	float fact; //!< Esta variable almacena el factor de incremento utilizado por la barra deslizante.

	float val;	//!< Este valor almacena el valor inicial de la barra deslizante.

	/*!
	   \fn barData(float f, float v)
	   \brief Constructor de la clase; inicializa los attributos fact y val.
	   \param f El valor con el que inicializamos el atributo fact.
	   \param v El valor con el que inicializamos el atributo val.
	*/
	barData(float f, float v)
	{
		fact = f;
		val = v;
	}
};

/*!
\fn void umLuzChange (int pos, void *data)
\brief Esta función es invocada por el GUI cada vez que el usuario interactua
con la barra deslizante asociada a un umbral de intensidad de luz.
\param pos Aquí se pasa la nueva posición de la barra
\param data Un apuntador a los datos asociados a la barra.

Esta función es invocada por el GUI cada vez que el usuario interactua con la
barra deslizante asociada a un umbral de intensidad de luz.
Cuando el usuario modifica la posición de la barra, se invoca esta fución, al
ser invocada se le pasa como parámetro la nueva posicion (con el parámetro pos),
y aun apuntador generico a datos que el usuario puede utilizar para modificar el
funcionamiento del programa (en nuestro caso una estructura del tipo barData).
*/
void umLuzChange(int pos, void* data)
{
	barData* umbral = (barData*)data;

	umbral->val = pos * umbral->fact;
}

/*!
\fn void umDistChange (int pos, void *data)
\brief Esta función es invocada por el GUI cada vez que el usuario interactua
con la barra deslizante asociada a un umbral de distancia.
\param pos Aquí se pasa la nueva posición de la barra
\param data Un apuntador a los datos asociados a la barra.

Esta función es invocada por el GUI cada vez que el usuario interactua con la
barra deslizante asociada a un umbral de distancia
Cuando el usuario modifica la posición de la barra, se invoca esta fución, al
ser invocada se le pasa como parámetro la nueva posicion (con el parámetro pos),
y aun apuntador generico a datos que el usuario puede utilizar para modificar el
funcionamiento del programa (en nuestro caso una estructura del tipo barData).
*/
void umDistChange(int pos, void* data)
{
	barData* umbral = (barData*)data;

	umbral->val = pow(pos * umbral->fact, 2);
	cout << "Umbral Dist :" << umbral->val << endl;
}

/*!
\fn void Umbraliza(Mat &Im, Mat &Mask, Mat &M, Mat &iCov, float umD, float umL)
\brief Esta funcion umbraliza la imagen a color en base a la distancia de Mahalanobis a un modelo de color.
\param Im Un objeto del tipo CV::Mat que reoresenta una imagen a color con tres
		  canales.
\param Mask Una referencia a un objeto del tipo CV::Mat en donde se regresa la
			máscara que indica que elementos de Im están arriba del umbral.
\param M Un objeto del tipo CV::Mat que contiene el vector promedio del modelo
	   de color con el cual se compara.
\param iCov Un objeto del tipo CV::Mat que contiene el inverso de la matriz de
			convarianza de modelo de color con el cual se compara.
\param umD El valor del umbral de distancia entre cada elemento procesado y el
		   modelo de color.
\param umL El valor del umbral de intensidad luminosa.

Esta funcion umbraliza la imagen a color en base a la distancia de Mahalanobis
a un modelo de color. El modelo de color esta determinado por el vector M, y la
matriz iCov, que almacenan el color promedio y la inversa de la matrix de
covarianza del modelo de color. El proceso de umbralización ocurre en dos
niveles: primero se descarta aquellos elementos cuya intensidad luminosa es
inferior al vlor umL, esto con el fin de eliminar pixeles obscuros, y segundo se
eliminan aquellos pixeles cuya distancia de Mahalanobis al modelo de color es
mayor el umD.

La imagen umbralizada se regresa en la matriz Mask.
*/
void Umbraliza(Mat& Im, Mat& Mask, Mat& M, Mat& iCov, float umD,
	float umL)
{
	Mat_ < Vec3f >::iterator it, itEnd;
	Mat_ < uchar >::iterator itM;
	float ligth, maha, ma, mb, va, vb, q, r, s;
	double meanMaha = 0;
	int cont = 0;

	it = Im.begin < Vec3f >();
	itEnd = Im.end < Vec3f >();
	itM = Mask.begin < uchar >();
	ma = M.at < float >(0, 0);
	mb = M.at < float >(1, 0);
	q = iCov.at < float >(0, 0);
	r = iCov.at < float >(1, 0);
	s = iCov.at < float >(1, 1);

	for (; it != itEnd; ++it, ++itM)
	{
		ligth = (*it)[0];					// We only analyze pixels that are bright enough.
		if (ligth > umL)
		{
			// We subtract the average vector from each pixel.
			va = (*it)[1] - ma;
			vb = (*it)[2] - mb;

			// We calculate the mahalanobis distance from the pixel to the model
			// [va,vb]*iCov*[va;vb]
			maha = vb * (s * vb + r * va) + va * (r * vb + q * va);
			meanMaha += maha;
			cont++;
			if (maha < umD)
				*itM = 255;
			else
				*itM = 0;
		}
		else
			*itM = 0;
	}
#ifdef __VERBOSE__
	if (cont)
		cout << "Mean Mahalanobis Distance: " << meanMaha << "/" << cont <<
		" : " << meanMaha / cont << endl;
#endif
}

std::vector<int> readTimes(string path)
{
	std::vector<int> data;
	std::ifstream file(path);

	if (!file)
		std::cerr << "File not found." << std::endl;
	else
	{
		int value;
		while (file >> value)
			data.push_back(value);
	}

	return data;
}

// Calculate the square root of a matrix.
cv::Mat sqrtMat(Mat matrix) 
{	
	//matrix = (n + lambda) * matrix;

	// Convert a Mat into a Eigen matrix using the constructor.
	Eigen::MatrixXf sqrtCovMatrix(matrix.rows, matrix.cols);

	// Pass data into the Eigen matrix.
	for (int i = 0; i < matrix.rows; i++)
		for (int j = 0; j < matrix.cols; j++)
			sqrtCovMatrix(i, j) = matrix.at<float>(i, j);

	// Calculate Cholesky decomposition.
	Eigen::LLT<Eigen::MatrixXf> lltOfA(sqrtCovMatrix);

	Eigen::MatrixXf L;
	// Check if the decomposition was successful.
	if (lltOfA.info() == Eigen::Success) {
		// Obtain the lower triangular matrix L from the Cholesky decomposition.
		L = lltOfA.matrixL();
	}
	else {
		std::cerr << "Cholesky's decomposition was not successful." << std::endl;
	}

	// Convert a matrix from Eigen library to opencv.
	Mat cvMatrix(L.rows(), L.cols(), CV_32F, L.data());

	return cvMatrix.t();
}

// Calculate the sigma points.
cv::Mat sigmaPointsUpdateState(Mat statePre, Mat sqrtmat, float theta)
{
	Mat sigmaPoints = cv::Mat::zeros(statePre.rows, 2 * statePre.rows + 1, CV_32F);

	sigmaPoints.col(0) = statePre;

	for (size_t i = 0; i < statePre.rows; i++)
	{
		sigmaPoints.col(i + 1) = statePre + theta * sqrtmat.col(i);
		sigmaPoints.col(i + 1 + statePre.rows) = statePre - theta * sqrtmat.col(i);
	}

	return sigmaPoints;
}

int main(int argc, char** argv)
{
	Mat frame, fFrame, labFrame, roi;
	Mat Mask, Mean, Cov, iCov, M;
	double iFact = 1. / 255.;
	bool firstImage;
	bool firstKalman;
	barData umDist(40. / SLIDE_MAX, 10);
	barData umLuz(100. / SLIDE_MAX, 0);
	int dSlidePos = 176, lSlidePos = 176;

	// Size of the state vector.
	int n = 6;

	// Initialization of Extended Kalman Filter
	KalmanFilter KF(n, 5, 0);

	std::vector<int> times = readTimes("Resorces/Data/TiemposNaranja.txt");

	// Camera calibration matrix
	Mat K =
		(Mat_ < float >(3, 3) <<
			7.7318146334666767e+02, 0.0, 4.0726293453767408e+02,
			0.0, 7.7318146334666767e+02, 3.0623163696686174e+02,
			0.0, 0.0, 1.0
			);

	// Inverse of camera calibration matrix.
	Mat KI;
	cv::invert(K, KI, cv::DECOMP_LU);

	int index = 1;
	float deltaT;
	float deltaTOld = 0;

	float k = 1;
	float alpha = 1;
	int beta = 2;
	float lambda = (alpha * alpha) * (n + k) - n;
	float theta = sqrt(n + lambda);

	float w0m = lambda / (n + lambda);
	float w0c = w0m + (1 - alpha * alpha + beta);
	float wi = 1 / (2 * (n + lambda));

	float X = 1;
	float Y = 1;
	float Z = 1;
	float XDer = 0;
	float YDer = 0;
	float ZDer = 0;

	float Rm = 0.0199;

	// Initialization of measurement vector
	Mat_ < float >measurement(5, 1);
	measurement.setTo(Scalar(0));

	// Initialize the stateVector
	KF.statePre.at < float >(0) = X;
	KF.statePre.at < float >(1) = Y;
	KF.statePre.at < float >(2) = Z;
	KF.statePre.at < float >(3) = XDer;
	KF.statePre.at < float >(4) = YDer;
	KF.statePre.at < float >(5) = ZDer;

	// Initialize the noise matrix
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1));
	setIdentity(KF.errorCovPost, Scalar::all(.1));

	// Read video
	//cv::VideoCapture cap("Resorces/Videos/GreeBallBlender25fpsTransparent.mkv");
	cv::VideoCapture cap("Resorces/Videos/PelotaNaranjaDayan.mkv");

	// check if we succeeded
	if (!cap.isOpened())
		return -1;

	// We define the size of the images to be captured
	cap.set(CAP_PROP_FRAME_WIDTH, IM_WIDTH);
	cap.set(CAP_PROP_FRAME_HEIGHT, IM_HEIGHT);

	namedWindow("Entrada", 1);
	namedWindow("Mascara", 1);

	createTrackbar("umDist", "Entrada", &dSlidePos, SLIDE_MAX, umDistChange, (void*)&umDist);
	createTrackbar("umLuz", "Entrada", &lSlidePos, SLIDE_MAX, umLuzChange, (void*)&umLuz);
	umDistChange(SLIDE_MAX, (void*)&umDist);
	umLuzChange(0, (void*)&umLuz);

	float w = 0.6;
	float sigma = 1;
	float p = 0.99;

	firstImage = true;
	firstKalman = true;

	float measurementXOld = measurement(0);
	float measurementYOld = measurement(1);

	Mat roiMask;
	Mat result;
	Rect roiRectCopy;
	Mat oldState = KF.statePost;

	int newSquareX = 0;
	int newSquareY = 0;
	int squareSize = 50;

	for (;;)
	{
		// We capture an image, and validate that the operation worked
		cap >> frame;
		if (frame.empty())
			break;

		if (!roiRectCopy.empty())
		{
			roiMask = cv::Mat::zeros(frame.size(), CV_8U);
			cv::rectangle(roiMask, roiRectCopy, cv::Scalar(255), FILLED);
			result = cv::Mat::zeros(frame.size(), CV_8U);
			frame.copyTo(result, roiMask);
			cv::imshow("Result", result);
			result.copyTo(frame);
		}

		frame.convertTo(fFrame, CV_32FC3);

		// It is necessary to normalize the BGR image to the interval [0,1] before converting to CIE Lab space
		// in this case iFact = 1./255
		fFrame *= iFact;
		cvtColor(fFrame, labFrame, COLOR_BGR2Lab);

		// In the first iteration we initialize the images that we will use to store results
		if (firstImage)
		{
			Size sz(frame.cols, frame.rows);

			Mat cFrame, mMask;

			// Calculates the color model based on the uploaded image
			// cFrame = imread (argv[2]);
			cFrame = imread("Resorces/Images/OrangeBallDayan.png");
			cFrame.convertTo(fFrame, CV_32FC3);
			fFrame *= iFact;
			cvtColor(fFrame, labFrame, COLOR_BGR2Lab);
			mMask = Mat::ones(cFrame.size(), CV_8UC1);
			MeaniCov(labFrame, mMask, Mean, iCov);

			Mask = Mat::ones(sz, CV_8U);

			firstImage = false;
		}

		cv::imshow("Entrada", frame);
		Umbraliza(labFrame, Mask, Mean, iCov, umDist.val, umLuz.val);

		// Find all contours in the image
		std::vector<std::vector<cv::Point>> contours;
		cv::imshow("Mascara", Mask);
		cv::findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat contourImage;
		frame.copyTo(contourImage);

		// Draw contours on image and show the resulting image
		cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 0, 255), 2);

		std::vector<std::vector<cv::Point>> filteredContours;
		Circle tempCircle;
		Circle bestCircle;

		float smallestError = 1000;

		for (const std::vector<cv::Point> contour : contours)
		{
			// Rule out contours with too few or too many points
			if (contour.size() >= 50)
			{
				std::vector<Point3s> fixedContour;
				unsigned int nInl;

				for (const cv::Point tempPoint : contour)
				{
					cv::Point3i newPoint(tempPoint.x, tempPoint.y, 0);
					fixedContour.push_back(newPoint);
				}

				// Apply RansacFit to find out if contour belongs to a circle or not
				tempCircle = Circle(fixedContour);
				float tempError = tempCircle.ransacFit(fixedContour, nInl, w, sigma, p);

				// Draw temp circle on countour image.
				if (tempError != -1)
					cv::circle(contourImage, cv::Point(tempCircle.h, tempCircle.k), tempCircle.r, cv::Scalar(0, 255, 0), 2);

				// Make shure that the contour saved is the one with lesser error
				if (tempCircle.r != 0 && tempError < smallestError)
				{
					bestCircle = tempCircle;
					smallestError = tempError;

					if (filteredContours.size() == 0)
					{
						filteredContours.push_back(contour);
					}
					else
					{
						filteredContours.pop_back();
						filteredContours.push_back(contour);
					}
				}
			}
		}

		cv::Mat filteredContourImage;
		frame.copyTo(filteredContourImage);
		cv::drawContours(filteredContourImage, filteredContours, -1, cv::Scalar(255, 0, 0), 2);

		if (bestCircle.r != 0)
		{
			// Draw square where we will look for the ball on the next iteration.
			newSquareX = bestCircle.h - bestCircle.r - (squareSize / 2);
			newSquareY = bestCircle.k - bestCircle.r - (squareSize / 2);

			if (newSquareX < 0)
				newSquareX = 0;

			if (newSquareY < 0)
				newSquareY = 0;

			cv::rectangle(filteredContourImage, Point(newSquareX, newSquareY),
				Point(newSquareX + bestCircle.r * 2 + squareSize, newSquareY + bestCircle.r * 2 + squareSize), Scalar(0, 255, 0), 2);

			cv::Rect roi_rect(newSquareX, newSquareY, squareSize + bestCircle.r * 2, squareSize + bestCircle.r * 2);
			roiRectCopy = roi_rect;

			if (!firstKalman)
			{
				// First predict, to update the internal statePre variable
				KF.statePre = KF.transitionMatrix * KF.statePost;
				KF.errorCovPre = KF.transitionMatrix * KF.errorCovPost * KF.transitionMatrix.t() + KF.processNoiseCov;

				X = KF.statePre.at < float >(0);
				Y = KF.statePre.at < float >(1);
				Z = KF.statePre.at < float >(2);
				XDer = KF.statePre.at < float >(3);
				YDer = KF.statePre.at < float >(4);
				ZDer = KF.statePre.at < float >(5);

				deltaT = times.at(index) - deltaTOld;
				deltaTOld = times.at(index);

				index++;

				Mat temp = KI * (Mat_ <float>(3, 1) << bestCircle.h, bestCircle.k, 1);
				measurement(0) = temp.at< float >(0, 0);
				measurement(1) = temp.at< float >(1, 0);
				measurement(2) = (measurementXOld - measurement(0)) / deltaT;
				measurement(3) = (measurementYOld - measurement(1)) / deltaT;
				measurement(4) = bestCircle.r * KI.at<float>(0, 0);

				measurementXOld = measurement(0);
				measurementYOld = measurement(1);

				// Calculate the square root of the error covariance matrix.
				Mat sqrtmat = sqrtMat(KF.errorCovPre);

				// Calculate the sigma points.
				Mat Sigmapoints = sigmaPointsUpdateState(KF.statePre, sqrtmat, theta);

				std::cout << Sigmapoints << std::endl;
			}
			else
			{
				deltaT = 40;
				deltaTOld = times.at(0);
				// Convert h and k from pixels to meters
				Mat temp = KI * (Mat_ <float>(3, 1) << bestCircle.h, bestCircle.k, 1);

				X = temp.at< float >(0, 0);
				Y = temp.at< float >(1, 0);
				Z = Rm / bestCircle.r;

				KF.statePre.at < float >(0) = X;
				KF.statePre.at < float >(1) = Y;
				KF.statePre.at < float >(2) = Z;
			}

			// Matrix A
			KF.transitionMatrix =
				(Mat_ < float >(6, 6) <<
					1, 0, 0, deltaT, 0, 0, \
					0, 1, 0, 0, deltaT, 0, \
					0, 0, 1, 0, 0, deltaT, \
					0, 0, 0, 1, 0, 0, \
					0, 0, 0, 0, 1, 0, \
					0, 0, 0, 0, 0, 1
					);

			// Matrix that relates de state vector with the measurement vector.
			Mat h =
				(Mat_ < float >(5, 1) <<
					X / Z,
					Y / Z,
					(XDer + ((X / Z) * ZDer)) / Z,
					(YDer + ((Y / Z) * ZDer)) / Z,
					Rm / Z
					);

			// Update the state from the last measurement.
			Mat temp = KF.measurementMatrix * KF.errorCovPre * KF.measurementMatrix.t() + KF.measurementNoiseCov;
			Mat inverse;
			cv::invert(temp, inverse, cv::DECOMP_LU);
			KF.gain = KF.errorCovPre * KF.measurementMatrix.t() * inverse;

			KF.statePost = KF.statePre + KF.gain * (measurement - h);
			KF.errorCovPost = (cv::Mat::eye(6, 6, CV_32F) - KF.gain * KF.measurementMatrix) * KF.errorCovPre;

			// Convert X, Y and r from state to pixels
			Mat tempDraw = K * (Mat_ <float>(3, 1) << KF.statePost.at<float>(0) / Z, KF.statePost.at<float>(1) / Z, 1);

			float drawH = tempDraw.at<float>(0, 0);
			float drawK = tempDraw.at<float>(1, 0);
			float drawR = K.at<float>(0, 0) * (Rm / KF.statePost.at<float>(2));

			try
			{
				cv::circle(filteredContourImage, cv::Point(drawH, drawK), drawR, cv::Scalar(0, 255, 0), 2);
			}
			catch (const std::exception&)
			{
				std::cout << "Could not draw the circle.";
			}

			firstKalman = false;
			oldState = KF.statePost;
		}

		cv::imshow("Countours", contourImage);
		cv::imshow("FilteredCountours", filteredContourImage);
		cv::imshow("Mascara", Mask);

		// If the user presses a key, the loop ends
		if (waitKeyEx(30) >= 0)
			break;
	}

	// Close windows that were opened
	cv::destroyWindow("Mascara");
	cv::destroyWindow("Entrada");

	return 0;
}
