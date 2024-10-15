#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <optional>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <TApplication.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TMultiGraph.h>

constexpr double LN = 1e-3; // Learning rate
constexpr int EPOCHS = 30;

/**
 * Split a string by a specified delimiter.
 *
 * This function takes a string and divides it into a vector of substrings
 * based on the given delimiter. It can be used to parse CSV or similar
 * formatted strings.
 *
 * @param [in] s The input string to be split.
 * @param [in] delimiter The character used as the delimiter to split the string.
 *
 * @return A vector of strings containing the substrings obtained by splitting
 *         the input string.
 */
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * Extract labels from the Iris dataset.
 *
 * This function assigns labels to samples in the dataset based on
 * whether they match the specified species name. Samples belonging
 * to the species will be labeled as -1, and all others will be labeled
 * as 1.
 *
 * @param [in] data A 2D vector containing the Iris dataset, where each
 *             inner vector represents a row with string values.
 *             The species name is expected to be in the fourth column.
 * @param [in] speciesName The species name to look for in the dataset.
 *             For example, "Iris-setosa".
 * @param [in] numRows The number of rows to process from the dataset.
 *                If set to -1, all rows will be processed.
 *                Defaults to -1.
 *
 * @return A vector of integers representing the labels assigned to each sample.
 *         The label will be -1 for the specified species and 1 for all others.
 */
std::vector<int> extractLabels(const std::vector<std::vector<std::string>>& data,
							   const std::string& speciesName, 
							   int numRows = -1) {
    std::vector<int> y;
    int rowsToProcess = (numRows == -1) ? data.size() : std::min(numRows, static_cast<int>(data.size()));
    
    for (int i = 0; i < rowsToProcess; ++i) {
        // iris labels can be found in 4th column
        if (data[i][4] == speciesName) { 
            y.push_back(-1);
        } else {
            y.push_back(1);
        }
    }
    return y;
}

/**
 * Extracts feature values from the given Iris dataset.
 *
 * This function retrieves the sepal length and petal length from
 * the provided dataset, converting them from strings to doubles.
 *
 * @param [in] data A 2D vector containing the Iris dataset, where each
 *             inner vector represents a row with string values.
 *             The expected format is that the first column is sepal
 *             length, and the third column is petal length.
 * @param [in] numRows The number of rows to process from the dataset.
 *                If set to -1, all rows will be processed.
 *                Defaults to -1.
 *
 * @return A 2D vector containing the extracted feature values.
 *         Each inner vector consists of two double values:
 *         [sepal length, petal length].
 */
std::vector<std::vector<double>> extractFeatures(const std::vector<std::vector<std::string>>& data, 
												 int numRows = -1) {
    std::vector<std::vector<double>> X;
    int rowsToProcess = (numRows == -1) ? data.size() : std::min(numRows, static_cast<int>(data.size()));

    for (int i = 0; i < rowsToProcess; ++i) {
        std::vector<double> row;
        row.push_back(std::stod(data[i][0]));  // Column 0: sepal length
        row.push_back(std::stod(data[i][2]));  // Column 2: petal length
        X.push_back(row);
    }
    return X;
}

using Layer = struct 
{
	std::vector<double> weights;
	double bias;
};


class Perceptron
{
private:
    double ln;
    int epochs;
	std::optional<int> seed;
	std::vector<double> errors; 
	Layer layer;
	std::default_random_engine generator;
  	std::normal_distribution<double> distribution;

public: 
	Perceptron(std::optional<int> random_state,
	           double learning_rate = LN,
			   int epochs_ = EPOCHS)
        : seed(random_state), ln(learning_rate), epochs(epochs_) {
    };
	Perceptron() {};
	~Perceptron() {};
	void fit(const std::vector<std::vector<double>>&, const std::vector<int>&);
	std::vector<double> net_input(const std::vector<std::vector<double>>&);
	std::vector<int> predict(const std::vector<std::vector<double>>&);
    int predict(const std::vector<double>& X);
	const std::vector<double>& getWeights() const {
        return layer.weights;  
    }
    const std::vector<double>& getErrors() const {
        return errors;  
    }
};

/**
 * @brief Train the perceptron model on the input data.
 * 
 * This function takes the feature matrix X and the corresponding target labels y, 
 * and trains the perceptron by adjusting its weights. The training process runs 
 * for a specified number of iterations (epochs), and in each iteration, the weights 
 * are updated based on the prediction error for each sample.
 * 
 * @param [in] X A 2D vector of feature values. Each inner vector represents the feature values for a sample.
 *          Rows represent samples and columns represent features.
 * @param [in] y A vector of target class labels corresponding to each sample in X. Each element of y is 
 *          either -1 or 1, representing the two classes.
 * 
 * @return void. The weights are updated internally after training.
 */
void Perceptron::fit(const std::vector<std::vector<double>>& X, 
					 const std::vector<int>& y)
{
	// Resize the weights to match the number of columns in X
	layer.weights.resize(X[0].size());

	// Initialize random number generator with the given random_state
    std::default_random_engine generator;
    // If random_state is provided, use it as a seed; otherwise, use std::random_device
    if (seed.has_value()) {
        generator.seed(seed.value());  // Use the provided seed
    } else {
        std::random_device rd;
        generator.seed(rd());  // Use non-deterministic seed
    }
    // Normal distribution with mean 0.0 and standard deviation 0.01
    std::normal_distribution<double> distribution(0.0, 0.01);
    
    // Fill weights with random values from the normal distribution
    for (int i = 0; i < layer.weights.size(); ++i) {
        layer.weights[i] = distribution(generator);
        	
    }
    // Initialize bias with a random value from the normal distribution
    layer.bias = distribution(generator);

    errors.resize(epochs);
    // For each training example :
    // a. Compute the output value (predict(X[j])
    // b. Update the weights
    for (int i = 0; i < epochs; ++i)
    {   
    	double errors_ = 0;
    	for (int j = 0; j < X.size(); ++j)
    	{
            // Perceptron learning rule
    		double update = ln * (y[j] - predict(X[j]));
            // Update each weight
            for (int k = 0; k < X[j].size(); ++k)
            {
                layer.weights[k] += update * X[j][k];
                
            }
            layer.bias += update;
            errors_ += (update != 0.0);
    	}
    	errors[i] = errors_;
    }
}

/**
 * Calculate the net input (weighted sum) for each sample in the dataset.
 *
 * @param [in] X A 2D vector.
 *             - Each inner vector is a feature vector for one sample.
 * @return A 1D vector of net inputs for each sample.
 *         - The net input is calculated as the dot product of the weights 
 *           and the input features plus the bias term.
 */
std::vector<double> Perceptron::net_input(const std::vector<std::vector<double>>& X)
{
	// Ensure that the number of columns in the matrix X equals the size of the vector weights
    if (X.empty() || X[0].size() != layer.weights.size()) {
        throw std::invalid_argument("Matrix columns must equal vector size.");
    }
    // compute the dot product
	std::vector<double> dot(X.size(), 0.0);
	for (int i = 0; i < X.size(); ++i)
	{
		for (int j = 0; j < X[i].size(); ++j)
			dot[i] += X[i][j] * layer.weights[j];
	}

	// add the bias
	std::vector<double> result(dot.size());
	for (int i = 0; i < dot.size(); ++i)
		result[i] = dot[i] + layer.bias;

	return result;
}

/**
 * Predict the class labels for each sample in the dataset.
 *
 * @param [in] X A 2D vector (input dataset).
 *             - Each inner vector is a feature vector for one sample.
 * @return A 1D vector of predicted class labels (1 or -1) for each sample.
 *
 */
std::vector<int> Perceptron::predict(const std::vector<std::vector<double>>& X)
{
	std::vector<double> activation_results = net_input(X);
	std::vector<int> result(X.size());
	for (int i = 0; i < activation_results.size(); ++i)
	{
		result[i] = (activation_results[i] >= 0.0) ? 1.0 : -1.0;
	}
	return result;
}

/**
 * Predict the class label for a single sample.
 *
 * @param [in] X A 1D vector representing the feature values for a single sample.
 * @return The predicted class label as an int (1 or -1).
 *
 */
int Perceptron::predict(const std::vector<double>& X) {
    std::vector<std::vector<double>> singleSample = { X }; // Wrap the 1D vector into a 2D vector
    std::vector<int> predictions = predict(singleSample);      
    return predictions[0]; // Return the first (and only) prediction
}

/**
 * Print a 2D vector of double values.
 *
 * This function outputs the contents of a 2D vector to the standard output.
 *
 * @param [in] vec A 2D vector containing double values to be printed.
 *
 */
void printVector(const std::vector<std::vector<double>>& vec) {
    for (const auto& row : vec) { 
        for (const auto& element : row) { 
            std::cout << element << " "; 
        }
        std::cout << std::endl; 
    }
}

int main(int argc, char** argv)
{    
    // Load iris dataset
	std::ifstream file("iris.csv");
    std::string line;
    std::vector<std::vector<std::string>> dataset;

    // Read the CSV file line by line
    while (std::getline(file, line)) {
        std::vector<std::string> row = split(line, ',');
        if (!row.empty()) {
            dataset.push_back(row);
        }
    }

    file.close();

  	// Define the species name
    std::string speciesName = "Iris-setosa";
    // Extract labels for first 100 rows
    std::vector<int> y = extractLabels(dataset, speciesName, 100);
    // Extract features for first 100 rows
    std::vector<std::vector<double>> X = extractFeatures(dataset, 100);

	auto seed = std::nullopt;
	
    // Create a Perceptron object and fit on data
	Perceptron nn(seed, LN, EPOCHS);
	nn.fit(X, y);

	// Check weights after fitting
    // std::cout << "Weights after fitting:" << std::endl;
    // for (const auto& weight : nn.getWeights()) {
    //     std::cout << weight << " ";
    // }
    // std::cout << std::endl;
    // // Check errors after fitting
    // const std::vector<double>& errors = nn.getErrors();
    // for (int i = 0; i < errors.size(); ++i)
    // {
    // 	std::cout << "errors: " << errors[i] << std::endl;
    // }
	
    // Create the application
    TApplication app("app", &argc, argv);
    // Create a canvas
    TCanvas *canvas = new TCanvas("canvas1", "Iris Dataset", 800, 600);
    
    // Create graphs for Setosa and Versicolor
    TGraph *graph_setosa = new TGraph();
    TGraph *graph_versicolor = new TGraph();

    int setosaIndex = 0;
    int versicolorIndex = 0;

    // Assuming y is populated correctly with -1 for Setosa and 1 for Versicolor
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] == -1) { // Setosa
            graph_setosa->SetPoint(setosaIndex++, X[i][0], X[i][1]);
        } else if (y[i] == 1) { // Versicolor
            graph_versicolor->SetPoint(versicolorIndex++, X[i][0], X[i][1]); 
        }
    }

    // Set marker styles and colors
    graph_setosa->SetMarkerStyle(20);
    graph_setosa->SetMarkerSize(1.);
    graph_setosa->SetMarkerColor(kRed);

    graph_versicolor->SetMarkerStyle(21);
    graph_versicolor->SetMarkerSize(1.);
    graph_versicolor->SetMarkerColor(kBlue);
   
    //std::cout << "Setosa points: " << graph_setosa->GetN() << std::endl;
    //std::cout << "Versicolor points: " << graph_versicolor->GetN() << std::endl;

    // Create a MultiGraph to combine both graphs
    TMultiGraph *mg = new TMultiGraph();
    mg->Add(graph_setosa);
    mg->Add(graph_versicolor);

    mg->SetTitle("Iris Dataset;Sepal Length (cm);Petal Length (cm)");

    // Draw the MultiGraph, which contains both graphs
    mg->Draw("AP");

    // Set axis labels (or use SetTitle)
    //mg->GetXaxis()->SetTitle("Sepal Length [cm]");
    //mg->GetYaxis()->SetTitle("Petal Length [cm]");

    // Create a legend
    TLegend *legend = new TLegend(0.1, 0.7, 0.3, 0.9);
    legend->AddEntry(graph_setosa, "Setosa", "p");
    legend->AddEntry(graph_versicolor, "Versicolor", "p");
    legend->Draw();

    canvas->Update();
    canvas->Modified();

    // Create a canvas
    TCanvas *canvas2 = new TCanvas("canvas2", "Predictions", 800, 600);

    // Predict on new test set
    std::vector<std::vector<double>> X_test = {
        {5.1, 3.5}, {4.9, 3.0}, {4.7, 3.2}, {4.6, 3.1}, {5.0, 3.6},
        {5.4, 3.9}, {4.6, 3.4}, {5.0, 3.4}, {6.4, 2.9}, {4.9, 3.1},
        {6.5, 3.0}, {6.6, 2.9}, {6.9, 3.1}, {6.3, 3.3}, {6.8, 2.8},
        {6.7, 3.0}, {6.1, 2.6}, {5.6, 2.8}, {5.7, 3.0}, {5.7, 2.8},
    };

    std::vector<int> preds;
    preds = nn.predict(X_test);

    // Create graphs for Setosa and Versicolor
    TGraph *graph_setosa2 = new TGraph();
    TGraph *graph_versicolor2 = new TGraph();

    int setosaIndex2 = 0;
    int versicolorIndex2 = 0;

    // Assuming y is populated correctly with -1 for Setosa and 1 for Versicolor
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == -1) { // Setosa
            graph_setosa2->SetPoint(setosaIndex2++, X_test[i][0], X_test[i][1]);
        } else if (preds[i] == 1) { // Versicolor
            graph_versicolor2->SetPoint(versicolorIndex2++, X_test[i][0], X_test[i][1]); 
        }
    }

    // Set marker styles and colors
    graph_setosa2->SetMarkerStyle(20);
    graph_setosa2->SetMarkerSize(1.);
    graph_setosa2->SetMarkerColor(kRed);

    graph_versicolor2->SetMarkerStyle(21);
    graph_versicolor2->SetMarkerSize(1.);
    graph_versicolor2->SetMarkerColor(kBlue);
   
    // Create a MultiGraph to combine both graphs
    TMultiGraph *mg2 = new TMultiGraph();

    // Only draw the Versicolor graph if it has points
    if (graph_versicolor2->GetN() > 0) {
        mg2->Add(graph_versicolor2);
    } else {
        std::cout << "No Versicolor points to plot." << std::endl;
    }
    // Only draw the Setosa graph if it has points
    if (graph_setosa2->GetN() > 0) {
        mg2->Add(graph_setosa2);
    } else {
        std::cout << "No Setosa points to plot." << std::endl;
    }

    mg2->SetTitle("Predictions on test data;Sepal Length (cm);Petal Length (cm)");

    // Draw the MultiGraph, which contains both graphs
    mg2->Draw("AP");

    // Create a legend
    TLegend *legend2 = new TLegend(0.1, 0.7, 0.3, 0.9);
    legend2->AddEntry(graph_setosa2, "Setosa", "p");
    legend2->AddEntry(graph_versicolor2, "Versicolor", "p");
    legend2->Draw();

    canvas2->Update();
    canvas2->Modified();

    app.Run();

    delete canvas;
    delete canvas2;

	return 0;
}

