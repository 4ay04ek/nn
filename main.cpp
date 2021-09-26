#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <algorithm>
#include <filesystem>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <SFML/Graphics.hpp>
 
using namespace cv;
using namespace boost;
using namespace std;
using namespace sf;

double sigmoid(double value) {
	return 1 / (1 + exp(-value));
}

double sigmoid_d(double value) {
	return value * (1 - value);
}

void init(string t, int *target, vector<string> &tmp) {
	for (int i = 0; i < 10; i++) target[i] = 0;
	target[t[t.size() - 1] - '0'] = 1;
	t.erase(t.end() - 3, t.end());
	erase_all(t, " ");
	split(tmp, t, is_any_of(","));
}

void randWeights(vector<vector<double> > &w) {
	for (auto &it: w) for (auto &j: it) {
		j = rand() / 500000.0;
	}
}

int main() {
	/*vector<pair<Mat, string> > data;
	for (int i = 0; i < 10; i++)
		for (auto& it : std::filesystem::directory_iterator("./data/MNIST Dataset JPG format/training/" + to_string(i))) {
			Mat img = imread(it.path().u8string(), IMREAD_GRAYSCALE);
			img = img.reshape(1, 1);
			data.push_back({ img, to_string(i) });
		}
	auto rng = default_random_engine{};
	shuffle(data.begin(), data.end(), rng);
	ofstream out("data.csv");
	for (auto it : data) out << format(it.first, Formatter::FMT_CSV) << ", " << it.second << '\n';
	out.close();*/
	double e = 0.0001, alpha = 0.00001;
	ifstream in("data.csv");
	const int i_n_c = 784, h_n_c = 7, o_n_c = 10;
	vector<vector<double> > h_weights(i_n_c, vector<double>(h_n_c)), o_weights(h_n_c, vector<double>(o_n_c));
	randWeights(h_weights);
	randWeights(o_weights); 
	string t;
	int k = 0;
	/*while (getline(in, t)) {
		int target[o_n_c]; 
		vector<string> tmp;
		init(t, target, tmp);
		double i_layer[i_n_c];
		double h_layer[h_n_c], o_layer[o_n_c];
		double delta_h_layer[h_n_c], delta_o_layer[o_n_c];
		double o_grad[h_n_c][o_n_c], h_grad[i_n_c][h_n_c];
		vector<vector<double> > pred_h_delta(i_n_c, vector<double>(h_n_c, 0)), pred_o_delta(h_n_c, vector<double>(o_n_c, 0));
		for (int i = 0; i < i_n_c; i++) {
			if (stoi(tmp[i]) > 127) i_layer[i] = 1;
			else i_layer[i] = 0;
		}
		for (int iters = 0; iters < 10000; iters++) {
			for (int i = 0; i < h_n_c; i++) {
				double sum = 0;
				for (int j = 0; j < i_n_c; j++) {
					sum += h_weights[j][i] * i_layer[j];
				}
				h_layer[i] = sigmoid(sum);
			}
			for (int i = 0; i < 10; i++) {
				double sum = 0;
				for (int j = 0; j < h_n_c; j++) {
					sum += o_weights[j][i] * h_layer[j];
				}
				o_layer[i] = sigmoid(sum);
			}
			for (int i = 0; i < o_n_c; i++) {
				delta_o_layer[i] = (target[i] - o_layer[i]) * sigmoid_d(o_layer[i]);
			}
			for (int i = 0; i < h_n_c; i++) {
				double sum = 0;
				for (int j = 0; j < o_n_c; j++) {
					sum += o_weights[i][j] * delta_o_layer[j];
				}
				delta_h_layer[i] = sigmoid_d(h_layer[i]) * sum;
			}
			for (int i = 0; i < h_n_c; i++) {
				for (int j = 0; j < o_n_c; j++) {
					o_grad[i][j] = h_layer[i] * delta_o_layer[j];
				}
			}
			for (int i = 0; i < i_n_c; i++) {
				for (int j = 0; j < h_n_c; j++) {
					h_grad[i][j] = i_layer[i] * delta_h_layer[j];
				}
			}
			for (int i = 0; i < i_n_c; i++) {
				for (int j = 0; j < h_n_c; j++) {
					double delta = e * h_grad[i][j] + alpha * pred_h_delta[i][j];
					pred_h_delta[i][j] = delta;
					h_weights[i][j] += delta;
				}
			}
			for (int i = 0; i < h_n_c; i++) {
				for (int j = 0; j < o_n_c; j++) {
					double delta = e * o_grad[i][j] + alpha * pred_o_delta[i][j];
					pred_o_delta[i][j] = delta;
					o_weights[i][j] += delta;
				}
			}
		}
		k++;
		printf("%d / 60000\n", k);
	}
	in.close();
	ofstream outw("weights.txt");
	for (int i = 0; i < i_n_c; i++) {
		for (int j = 0; j < h_n_c; j++) {
			outw << h_weights[i][j] << ' ';
		}
		outw << '\n';
	}
	for (int i = 0; i < h_n_c; i++) {
		for (int j = 0; j < o_n_c; j++) {
			outw << o_weights[i][j] << ' ';
		}
		outw << '\n';
	}
	outw.close();*/
	in.close();
	ifstream inw("weights.txt");
	for (int i = 0; i < i_n_c; i++) {
		for (int j = 0; j < h_n_c; j++) {
			inw >> h_weights[i][j];
		}
	}
	for (int i = 0; i < h_n_c; i++) {
		for (int j = 0; j < o_n_c; j++) {
			inw >> o_weights[i][j];
		}
	}
	inw.close();
	/*ofstream out("test.csv");
	for (int i = 0; i < 10; i++)
		for (auto& it : std::filesystem::directory_iterator("./data/MNIST Dataset JPG format/testing/" + to_string(i))) {
			Mat img = imread(it.path().u8string(), IMREAD_GRAYSCALE);
			img = img.reshape(1, 1);
			out << format(img, Formatter::FMT_CSV) << ", " << to_string(i) << '\n';
		}
	out.close();*/
	/*in.open("test.csv");
	while (getline(in, t)) {
		vector<string> tmp;
		char target = t[t.size() - 1];
		t.erase(t.end() - 3, t.end());
		erase_all(t, " ");
		split(tmp, t, is_any_of(","));
		double i_layer[i_n_c];
		double h_layer[h_n_c], o_layer[o_n_c];
		for (int i = 0; i < i_n_c; i++) {
			if (stoi(tmp[i]) > 127) i_layer[i] = 1;
			else i_layer[i] = 0;
		}
		for (int i = 0; i < h_n_c; i++) {
			double sum = 0;
			for (int j = 0; j < i_n_c; j++) {
				sum += h_weights[j][i] * i_layer[j];
			}
			h_layer[i] = sigmoid(sum);
		}
		for (int i = 0; i < 10; i++) {
			double sum = 0;
			for (int j = 0; j < h_n_c; j++) {
				sum += o_weights[j][i] * h_layer[j];
			}
			o_layer[i] = sigmoid(sum);
		}
		for (int i = 0; i < 10; i++) printf("%0.2f ", o_layer[i]);
		cout << target << '\n';
	}*/
	RenderWindow wnd(VideoMode(520, 420), "GUI");
	vector<int> input(28 * 28);
	vector<vector<int> > quad_input(28, vector<int>(28, 0));
	Font font;
	if (!font.loadFromFile("arial.ttf")){}
	while (wnd.isOpen())
	{
		Event event;
		wnd.clear(Color::White);
		while (wnd.pollEvent(event))
		{
			if (event.type == Event::Closed)
				wnd.close();
		}
		if (Mouse::isButtonPressed(Mouse::Left))
		{
			Vector2i pos = Mouse::getPosition(wnd);
			if (pos.x < 420 && pos.y < 420 && pos.x > 0 && pos.y > 0) {
				quad_input[pos.y / 15][pos.x / 15] = 1;
			}
		}
		input.clear();
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				input.push_back(quad_input[i][j]);
				if (quad_input[i][j]) {
					RectangleShape rect(Vector2f(15, 15));
					rect.setFillColor(Color::Black);
					rect.setPosition(Vector2f(j * 15, i * 15));
					wnd.draw(rect);
				}
			}
		}
		double h_layer[h_n_c], o_layer[o_n_c];
		for (int i = 0; i < h_n_c; i++) {
			double sum = 0;
			for (int j = 0; j < i_n_c; j++) {
				sum += h_weights[j][i] * input[j];
			}
			h_layer[i] = sigmoid(sum);
		}
		for (int i = 0; i < o_n_c; i++) {
			double sum = 0;
			for (int j = 0; j < h_n_c; j++) {
				sum += o_weights[j][i] * h_layer[j];
			}
			o_layer[i] = sigmoid(sum);
		}
		for (int i = 0; i < 10; i++) {
			printf("%0.2f ", o_layer[i]);
			RectangleShape rect(Vector2f(5, 50 * o_layer[i]));
			rect.setFillColor(Color::Red);
			rect.setPosition(Vector2f(430, 10 + 42 * i));
			Text num(to_string(i), font, 16);
			num.setPosition(Vector2f(450, 20 + 42 * i));
			num.setFillColor(Color::Black);
			wnd.draw(rect);
			wnd.draw(num);
		}
		cout << '\n';
		wnd.display();
	}
}