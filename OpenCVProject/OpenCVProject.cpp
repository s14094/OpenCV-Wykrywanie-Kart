#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

int thresh = 50, N = 5;
const char* wndname = "Wykrywanie kart";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares(const Mat& image, vector<vector<Point> >& squares)
{
	squares.clear();

	//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

	// down-scale and upscale the image to filter out the noise
	//pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
	//pyrUp(pyr, timg, image.size());


	// blur will enhance edge detection
	Mat timg(image);
	medianBlur(image, timg, 9);
	Mat gray0(timg.size(), CV_8U), gray;

	vector<vector<Point> > contours;

	// find squares in every color plane of the image
	for (int c = 0; c < 3; c++)
	{
		int ch[] = { c, 0 };
		mixChannels(&timg, 1, &gray0, 1, ch, 1);

		// try several threshold levels
		for (int l = 0; l < N; l++)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. Take the upper threshold from slider
				// and set the lower to 0 (which forces edges merging)
				Canny(gray0, gray, 5, thresh, 5);
				// dilate canny output to remove potential
				// holes between edge segments
				dilate(gray, gray, Mat(), Point(-1, -1));
			}
			else
			{
				// apply threshold if l!=0:
				//     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				gray = gray0 >= (l + 1) * 255 / N;
			}

			// find contours and store them all as a list
			findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

			vector<Point> approx;

			// test each contour
			for (size_t i = 0; i < contours.size(); i++)
			{
				// approximate contour with accuracy proportional
				// to the contour perimeter
				approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

				// square contours should have 4 vertices after approximation
				// relatively large area (to filter out noisy contours)
				// and be convex.
				// Note: absolute value of an area is used because
				// area may be positive or negative - in accordance with the
				// contour orientation
				if (approx.size() == 4 &&
					fabs(contourArea(Mat(approx))) > 1000 &&
					isContourConvex(Mat(approx)))
				{
					double maxCosine = 0;

					for (int j = 2; j < 5; j++)
					{
						// find the maximum cosine of the angle between joint edges
						double cosine = fabs(angle(approx[j % 4], approx[j - 2], approx[j - 1]));
						maxCosine = MAX(maxCosine, cosine);
					}

					// if cosines of all angles are small
					// (all angles are ~90 degree) then write quandrange
					// vertices to resultant sequence
					if (maxCosine < 0.3)
						squares.push_back(approx);
				}
			}
		}
	}
}

void checkSquaresPositions(const vector<vector<Point> >& squares, bool &LT, bool &LM, bool &LB, bool &RT, bool &RM, bool &RB)
{


	system("cls");
	for (size_t i = 0; i < squares.size(); i++)
	{
		auto p1 = &squares[i][0];
		auto p2 = &squares[i][1];
		auto p3 = &squares[i][2];
		auto p4 = &squares[i][3];
		int x1 = p1->x;
		int y1 = p1->y;
		int x2 = p2->x;
		int y2 = p2->y;
		int x3 = p3->x;
		int y3 = p3->y;
		int x4 = p4->x;
		int y4 = p4->y;

		//left
		if ((x1 > 200) && (x2 > 200) && (x3 > 200) && (x4 > 200) && (x1 < 322) && (x2 < 322) && (x3 < 322) && (x4 < 322))
		{
			//top
			if ((y1 > 80) && (y2 > 80) && (y3 > 80) && (y4 > 80) && (y1 < 200) && (y2 < 200) && (y3 < 200) && (y4 < 200))
			{
				LT = true;
			}
			//middle
			else if ((y1 > 200) && (y2 > 200) && (y3 > 200) && (y4 > 200) && (y1 < 300) && (y2 < 300) && (y3 < 300) && (y4 < 300))
			{
				LM = true;
			}
			//bottom
			else if ((y1 > 300) && (y2 > 300) && (y3 > 300) && (y4 > 300) && (y1 < 420) && (y2 < 420) && (y3 < 420) && (y4 < 420))
			{
				LB = true;
			}
		}

		//right
		if ((x1 > 322) && (x2 > 322) && (x3 > 322) && (x4 > 322) && (x1 < 450) && (x2 < 450) && (x3 < 450) && (x4 < 450))
		{
			//top
			if ((y1 > 80) && (y2 > 80) && (y3 > 80) && (y4 > 80) && (y1 < 200) && (y2 < 200) && (y3 < 200) && (y4 < 200))
			{
				RT = true;
			}
			//middle
			else if ((y1 > 200) && (y2 > 200) && (y3 > 200) && (y4 > 200) && (y1 < 300) && (y2 < 300) && (y3 < 300) && (y4 < 300))
			{
				RM = true;
			}
			//bottom
			else if ((y1 > 300) && (y2 > 300) && (y3 > 300) && (y4 > 300) && (y1 < 420) && (y2 < 420) && (y3 < 420) && (y4 < 420))
			{
				RB = true;
			}
		}
	}
}

void checkCards(bool LT, bool LM, bool LB, bool RT, bool RM, bool RB, int &suit, int &card, char &currentCard, char &currentSuit)
{
	//pik
	if (LT == false && RT == false)
	{
		suit = 1;
		currentSuit = char(006);
	}
	//trefl
	if (LT == true && RT == false)
	{
		suit = 2;
		currentSuit = char(005);
	}
	// karo
	if (LT == false && RT == true)
	{
		suit = 3;
		currentSuit = char(004);
	}
	//kier
	if (LT == true && RT == true)
	{
		suit = 4;
		currentSuit = char(003);
	}
	// A
	if (LM == false && RM == false && LB == false && RB == true)
	{
		card = 1;
		currentCard = 'A';
	}
	// 2
	if (LM == false && RM == false && LB == true && RB == false)
	{
		card = 2;
		currentCard = '2';
	}
	// 3
	if (LM == false && RM == false && LB == true && RB == true)
	{
		card = 3;
		currentCard = '3';
	}
	// 4
	if (LM == false && RM == true && LB == false && RB == false)
	{
		card = 4;
		currentCard = '4';
	}
	// 5
	if (LM == false && RM == true && LB == false && RB == true)
	{
		card = 5;
		currentCard = '5';
	}
	// 6
	if (LM == false && RM == true && LB == true && RB == false)
	{
		card = 6;
		currentCard = '6';
	}
	// 7
	if (LM == false && RM == true && LB == true && RB == true)
	{
		card = 7;
		currentCard = '7';
	}
	// 8
	if (LM == true && RM == false && LB == false && RB == false)
	{
		card = 8;
		currentCard = '8';
	}
	// 9
	if (LM == true && RM == false && LB == false && RB == true)
	{
		card = 9;
		currentCard = '9';
	}
	// 10
	if (LM == true && RM == false && LB == true && RB == false)
	{
		card = 10;
		currentCard = '1';
	}
	// J
	if (LM == true && RM == false && LB == true && RB == true)
	{
		card = 11;
		currentCard = 'J';
	}
	// D
	if (LM == true && RM == true && LB == false && RB == false)
	{
		card = 12;
		currentCard = 'D';
	}
	// K
	if (LM == true && RM == true && LB == false && RB == true)
	{
		card = 13;
		currentCard = 'K';
	}
	// poza
	if (LM == true && RM == true && LB == true && RB == false)
	{
		cout << "brak" << endl;
	}
	// poza
	if (LM == true && RM == true && LB == true && RB == true)
	{
		cout << "brak" << endl;
	}

}

void drawCards(char currentCard, char currentSuit)
{
	const int cardHeight = 14;
	const int cardWidth = 14;
	char tenT = ' ';
	char tenB = ' ';

	system("cls");
	char cardView[cardHeight][cardWidth] = {
		"1555555555552",
		"6           6",
		"6 cd      S 6",
		"6           6",
		"6           6",
		"6           6",
		"6           6",
		"6           6",
		"6           6",
		"6           6",
		"6           6",
		"6 S      DC 6",
		"6           6",
		"3555555555554" };

	if (currentSuit == char(006) || currentSuit == char(005))
	{
		system("color F0");
	}
	if (currentSuit == char(004) || currentSuit == char(003))
	{
		system("color F4");
	}
	for (int y = 0; y < cardHeight; ++y) {
		for (int x = 0; x < cardWidth; ++x) {
			char curChar = cardView[y][x];
			switch (curChar) {
			case '1':
				cardView[y][x] = char(201);
				break;
			case '2':
				cardView[y][x] = char(187);
				break;
			case '3':
				cardView[y][x] = char(200);
				break;
			case '4':
				cardView[y][x] = char(188);
				break;
			case '5':
				cardView[y][x] = char(205);
				break;
			case '6':
				cardView[y][x] = char(186);
				break;
				//dol
			case 'C':
				if (currentCard == '1')
				{
					tenB = '0';
					cardView[y][x] = tenB;
				}
				else
				{
					cardView[y][x] = currentCard;
				}
				break;
			case 'D':
				if (tenT == '0')
				{
					cardView[y][x] = '1';
				}
				else
				{
					cardView[y][x] = tenB;
				}
				break;
				//gora
			case 'c':
				if (currentCard == '1')
				{
					cardView[y][x] = currentCard;
					tenT = '0';
				}
				else
				{
					cardView[y][x] = currentCard;
				}
				break;
			case 'd':
				cardView[y][x] = tenT;
				break;
			case 'S':
				cardView[y][x] = currentSuit;
				break;
			default:
				break;
			}
			cout << cardView[y][x];
		}
		cout << endl;
	}

}

/*
void createDeck(string deck[])
{
const int numSuit = 4;
const int numCard = 13;

string suits[numSuit] = { "pik", "trefl", "karo", "kier" };
string cards[numCard] = { "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "D", "K" };
int index = 0;
for (int i = 0; i < numSuit; i++)
{
for (int j = 0; j < numCard; j++)
{
deck[index++] = suits[i] + " " + cards[j];
}
}
}
*/

// the function draws all the squares in the image
static void drawSquares(Mat& image, const vector<vector<Point> >& squares)
{
	for (size_t i = 0; i < squares.size(); i++)
	{
		const Point* p = &squares[i][0];

		int n = static_cast<int>(squares[i].size());
		//dont detect the border
		if (p->x > 3 && p->y > 3)
			polylines(image, &p, &n, 1, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
	imshow(wndname, image);
}

void interface(Mat image)
{
	Scalar lineColor = Scalar(0, 255, 0);
	line(image, Point(322, 100), Point(322, 400), lineColor, 3); //pionowa linia
	line(image, Point(222, 200), Point(422, 200), lineColor, 3); //pozioma gorna
	line(image, Point(222, 300), Point(422, 300), lineColor, 3); //pozioma dolna
}

/*
source https://github.com/alyssaq/opencv/blob/master/squares.cpp
*/
// jezeli nie wyswietla znakow, console - otions - use old console
// zeby lepiej wyswietlalo zmiana konsoli na Lucida Console.

int main(int /*argc*/, char** /*argv*/)
{
	int suit = 0;
	int card = 0;
	char currentCard;
	char currentSuit;
	namedWindow(wndname, 1);
	vector<vector<Point> > squares;
	bool LT, LM, LB, RT, RM, RB;
	bool continueCapture = true;
	VideoCapture cap(0);

	//string deck[52];
	//createDeck(deck);

	if (!cap.isOpened())
	{
		return -1;
	}

	while (continueCapture) {
		Mat image;
		cap >> image;
		if (cap.read(image))
		{
			LT = false, LM = false, LB = false, RT = false, RM = false, RB = false;
			findSquares(image, squares);
			interface(image);
			checkSquaresPositions(squares, LT, LM, LB, RT, RM, RB);
			checkCards(LT, LM, LB, RT, RM, RB, suit, card, currentCard, currentSuit);
			drawCards(currentCard, currentSuit);
			drawSquares(image, squares);
			auto c = waitKey();
			if (static_cast<char>(c) == 27)
			{
				return 0;
			}
			if (char(c) == 67 || char(c) == 99)
			{
				// zapisywanie danej karty
				cout << "asdasddsa" << endl;
			}
			currentCard = ' ';
			currentSuit = ' ';
		}
		else continueCapture = false;
	}
	return 0;
}