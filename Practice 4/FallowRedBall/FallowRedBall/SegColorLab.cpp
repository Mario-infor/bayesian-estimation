#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <cstdio>
#include "Circle.h"
#include <cmath>

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
		ligth = (*it)[0];					//Solo analizamos pixeles lo suf. brillantes.
		if (ligth > umL)
		{
			// Restamos el vector promedio a cada pixel.
			va = (*it)[1] - ma;
			vb = (*it)[2] - mb;

			//Calculamos la distancia de mahalanobis del pixel al modelo
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

int main(int argc, char** argv)
{
	Mat frame, fFrame, labFrame;
	Mat Mask, Mean, Cov, iCov, M;
	//int cont;
	double iFact = 1. / 255.;
	bool first;
	barData umDist(40. / SLIDE_MAX, 10);
	barData umLuz(100. / SLIDE_MAX, 0);
	int dSlidePos = 16, lSlidePos = 16;



	/*if (argc < 2)
	{
		   cerr << "Faltan Parámetros." << endl;
		   cerr << "Uso: SegColorLab NoCamara modelo_color" << endl << endl;
		   exit (1);
	}*/

	//VideoCapture cap (atoi (argv[1]));	// open the default camera

	cv::VideoCapture cap("Resorces/Videos/PelotaVerde.mkv");

	if (!cap.isOpened())				// check if we succeeded
		return -1;

	//Definimos el tamaño de las imagenes a ser capturadas.
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

	first = true;
	for (;;)
	{
		//Capturamos una imagen, y validamos que haya funcionado la operacion.
		cap >> frame;
		if (frame.empty())
			break;

		frame.convertTo(fFrame, CV_32FC3);

		//Es necesario normalizar la image BGR al intervalo [0,1] antes de convertir a espacio CIE Lab; en este caso iFact = 1./255
		fFrame *= iFact;
		cvtColor(fFrame, labFrame, COLOR_BGR2Lab);

		//En la primera iteración inicializamos las imagenes que usaremos para
		//almacenar resultados.
		if (first)
		{
			Size sz(frame.cols, frame.rows);

			Mat cFrame, mMask;

			//Calcula el modelo de color en base a la imagen cargada.
			//cFrame = imread (argv[2]);
			cFrame = imread("Resorces/Images/ColorVerde.png");
			cFrame.convertTo(fFrame, CV_32FC3);
			fFrame *= iFact;
			cvtColor(fFrame, labFrame, COLOR_BGR2Lab);
			mMask = Mat::ones(cFrame.size(), CV_8UC1);
			MeaniCov(labFrame, mMask, Mean, iCov);

			Mask = Mat::ones(sz, CV_8U);

			first = false;
		}

		imshow("Entrada", frame);

		Umbraliza(labFrame, Mask, Mean, iCov, umDist.val, umLuz.val);


		/*Aqui tienen que hacer lo necesario para que a la imagen segmentada
		que está almacenada en Mask se extraga el contorno y se ajuste un
		circulo a esta. Para esto, usar la función findContours.

		Algo asó como lo que sigue:

		findContours(output, Contornos, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		Donde Contorns es como sigue:

		vector< vector<Point> >Contornos;*/

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(Mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat contourImage = cv::Mat::zeros(Mask.size(), CV_8UC3);
		cv::drawContours(contourImage, contours, -1, cv::Scalar(0, 0, 255), 2);

		imshow("Countours", contourImage);

		/*Cada elemento de Contornos es una vector de coordenadas y cada uno
		de estos vectores corresponde a un objeto segmentado en la imagen.
		Estos vectores hay que ajustarlos a un circulo usando la clase
		Circle y ransacFit.*/

		std::vector<std::vector<cv::Point>> filteredcontours;

		float smallestError = 1000;

		for (const std::vector<cv::Point> contour : contours)
		{
			if (contour.size() >= 150 && contour.size() <= 350)
			{
				std::vector<Point3s> fixedContour;
				unsigned int nInl;

				for (const cv::Point tempPoint : contour)
				{
					cv::Point3i newPoint(tempPoint.x, tempPoint.y, 0);
					fixedContour.push_back(newPoint);
				}

				Circle tempCircle = Circle(fixedContour);

				//float error = tempCircle.fitCircle(fixedContour, w, sigma, p);
				float tempError = tempCircle.ransacFit(fixedContour, nInl, w, sigma, p);

				if (tempError < smallestError && tempError < 4)
				{
					
					if (filteredcontours.size() == 0) 
					{
						filteredcontours.push_back(contour);
					}
					else
					{
						filteredcontours.pop_back();
						filteredcontours.push_back(contour);
					}
					smallestError = tempError;
				}
			}
		}

		cv::Mat filteredContourImage = cv::Mat::zeros(Mask.size(), CV_8UC3);
		cv::drawContours(filteredContourImage, filteredcontours, -1, cv::Scalar(255, 0, 0), 2);

		imshow("FilteredCountours", filteredContourImage);

		imshow("Mascara", Mask);

		//Si el usuario oprime una tecla, termina el ciclo.
		if (waitKeyEx(30) >= 0)
			break;

	}

	cv::imwrite("LastFrame.png", frame);

	//Cierra ventanas que fueron abiertas.
	cv::destroyWindow("Mascara");
	cv::destroyWindow("Entrada");

	return 0;
}
