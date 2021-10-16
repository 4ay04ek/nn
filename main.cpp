#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <random>
#include <algorithm>
#include <filesystem>
#include <windows.h>
#include <armadillo>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>

#define ITERS 10
#define EPOCHS 3
#define lr 0.1

using namespace boost;
using namespace std;
using namespace sf;
using namespace arma;

mat cv2arma(string path) {
	cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
	vector<double> t;
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++) {
			t.push_back((int)image.at<uchar>(i, j) / 255.0);
		}
	mat input(t);
	return input.t();
}

void sigmoid(double &value) {
	value = 1 / (1 + exp(-value));
}

void sigmoid_d(double &value) {
	sigmoid(value);
	value = value * (1 - value);
}

mat cost(mat t, mat p) {
	return 0.5 * pow(p - t, 2);
}

mat cost_d(mat t, mat p) {
	return p - t;
}

double error_rate(mat o, mat t) {
	mat r = cost_d(t, o);
	return accu(r) / 10;
}

void forward(mat i, mat hw, mat& h, mat ow, mat& o, mat& zo, mat& zh) {
	zh = mat(i.t() * hw).t();
	h = zh; h.for_each(sigmoid);
	zh.for_each(sigmoid_d);
	zo = mat(h.t() * ow).t();
	o = zo; o.for_each(sigmoid);
	zo.for_each(sigmoid_d);
}

void back(mat i, mat t, mat& hw, mat h, mat& ow, mat o, mat zo, mat zh) {
	mat eO = cost_d(t, o) % zo;
	mat eH = ow * eO % zh;
	mat dow = h * eO.t();
	mat dhw = i * eH.t();
	hw -= lr * dhw;
	ow -= lr * dow;
}


int main() {
	HANDLE hd = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info;
	info.dwSize = 100;
	info.bVisible = FALSE;
	SetConsoleCursorInfo(hd, &info);
	string train_dir = "C:/Users/html_programmer/source/repos/nn/data/MNIST Dataset JPG format/training";
	string test_dir = "C:/Users/html_programmer/source/repos/nn/data/MNIST Dataset JPG format/testing";
	arma_rng::set_seed_random();
	mat hw(784, 28), ow(28, 10);
	mt19937 engine;
	uniform_real_distribution<double> distr(-1, 1);
	hw.imbue([&]() { return distr(engine); });
	ow.imbue([&]() { return distr(engine); });
	vector<pair<string, int> > train_data, test_data;
	for (int target = 0; target < 10; target++) {
		for (auto it: filesystem::directory_iterator(train_dir + "/" + to_string(target))) {
			train_data.push_back({ it.path().u8string(), target });
		}
	}
	shuffle(train_data.begin(), train_data.end(), engine);
	int k = 0;
	for (int epoch = 0; epoch < EPOCHS; epoch++, k = 0)
		for (auto data : train_data) {
			vector<double> vt(10, 0); vt[data.second] = 1;
			mat i = cv2arma(data.first).t();
			mat h(28, 1), o(10, 1), t(vt), zh(28, 1), zo(10, 1);
			for (int iters = 0; iters < ITERS; iters++) {
				forward(i, hw, h, ow, o, zh, zo);
				back(i, t, hw, h, ow, o, zh, zo);
			}
			//cout << fixed;
			//cout << "Target: " << data.second << "-------->" << error_rate(o, t) << " % <--------\n";
			//o.t().raw_print();
			k++;
			printf("Epoch %d: %d/60000\n", epoch + 1, k);
			COORD c; c.X = 0; c.Y = 0;
			SetConsoleCursorPosition(hd, c);
		}
	for (int target = 0; target < 10; target++) {
		for (auto it : filesystem::directory_iterator(test_dir + "/" + to_string(target))) {
			test_data.push_back({ it.path().u8string(), target });
		}
	}
	for (auto data : test_data) {
		vector<double> vt(10, 0); vt[data.second] = 1;
		mat i = cv2arma(data.first).t();
		mat h(28, 1), o(10, 1), t(vt), zh(28, 1), zo(10, 1);
		for (int iters = 0; iters < ITERS; iters++) {
			forward(i, hw, h, ow, o, zh, zo);
		}
		cout << fixed;
		cout.precision(3);
		cout << "Target: " << data.second << "-------->" << error_rate(o, t) << " % <--------\n";
		o.t().raw_print();
		COORD c; c.X = 0; c.Y = 0;
		SetConsoleCursorPosition(hd, c);
	}
	RenderWindow wnd(VideoMode(520, 460), "GUI");
	vector<double> input(28 * 28);
	vector<vector<double> > quad_input(28, vector<double>(28, 0));
	RectangleShape button(Vector2f(520, 40));
	button.setFillColor(Color(150, 150, 150));
	Font font;
	string input_path;
	if (!font.loadFromFile("arial.ttf")) {}
	while (wnd.isOpen())
	{
		Event event;
		wnd.clear(Color::White);
		while (wnd.pollEvent(event))
		{
			if (event.type == Event::Closed)
				wnd.close();
			if (event.type == Event::KeyPressed) {
				if (event.key.code == Keyboard::Space) {
					std::fill(quad_input.begin(), quad_input.end(), vector<double>(28, 0));
				}
			}
		}
		if (Mouse::isButtonPressed(Mouse::Left))
		{
			Vector2i pos = Mouse::getPosition(wnd);
			if (pos.x < 420 && pos.y < 460 && pos.x > 0 && pos.y > 40) {
				quad_input[(pos.y - 40) / 15][pos.x / 15] = 1;
			}
			if (button.getGlobalBounds().contains((Vector2f)pos)) {
				system("cls");
				getline(cin, input_path);
				cv::Mat image = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
				for (int i = 0; i < image.rows; i++)
					for (int j = 0; j < image.cols; j++) {
						quad_input[i][j] = (int)image.at<uchar>(i, j) / 255.0;
					}
			}
		}
		input.clear();
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				input.push_back(quad_input[i][j]);
				if (quad_input[i][j] != 0) {
					RectangleShape rect(Vector2f(15, 15));
					rect.setFillColor(Color(255 - 255 * quad_input[i][j], 255 - 255 * quad_input[i][j], 255 - 255 * quad_input[i][j]));
					rect.setPosition(Vector2f(j * 15, i * 15 + 40));
					wnd.draw(rect);
				}
			}
		}
		mat in(input);
		mat h(28, 1), o(10, 1), zh(28, 1), zo(10, 1);
		forward(in, hw, h, ow, o, zh, zo);
		for (int i = 0; i < 10; i++) {
			RectangleShape rect(Vector2f(5, 50 * o.at(i)));
			rect.setFillColor(Color::Red);
			rect.setPosition(Vector2f(430, 50 + 42 * i));
			Text num(to_string(i), font, 16);
			num.setPosition(Vector2f(450, 60 + 42 * i));
			num.setFillColor(Color::Black);
			wnd.draw(rect);
			wnd.draw(num);
		}
		wnd.draw(button);
		wnd.display();
	}
}