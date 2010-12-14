/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#include "classifier.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>

using std::string;

/// command-line params
struct params {
    typedef std::vector<string> StringList;

    void(*mode)(params&);  // mode function pointer
    string prog;           // program name
    string out_file;       // neural net file
    string cls_file;       // classifier filename
    string data_dir;       // data dir to process
    StringList files;      // classification files
    StringList labels;     // data labels
    bool verbose;          // verbosity
    size_t hidden_neurons; // number of hidden neurons
    size_t chunk_size;     // size of BP learning chunks
    size_t try_count;      // how many NNs to train to choose the best one
};

/// write program help to stdout
void prog_help(params& p)
{
    std::cout << p.prog << " <mode> <switches>\n"
              "    mode is one of: train, classify, dataset, test, help\n"
              "\n"
              "\n"
              << std::endl;
}

/// process data set
void do_dataset(params& p)
{
    if (p.labels.size() == 0)  throw std::runtime_error("Specify desired labels.");
    if (p.data_dir.empty())    throw std::runtime_error("Specify data directory.");
    if (p.out_file.empty())    throw std::runtime_error("Specify output filename.");

    DataSet data;
    std::cout << "=== Loading data & extracting features" << std::endl;
    data.load_dir(p.data_dir, p.labels);
    std::cout << "=== Normalizing data" << std::endl;
    data.normalize_all();
    std::cout << "=== Shuffling data" << std::endl;
    data.shuffle();
    std::cout << "=== Writing data" << std::endl;
    data.write_tmp(p.out_file);
    std::cout << "=== DONE" << std::endl;
}

/// training
void training(params& p)
{
    if (p.hidden_neurons == 0) throw std::runtime_error("Specify number of hidden layser neurons.");
    if (p.out_file.empty())    throw std::runtime_error("Specify classifier output filename.");
    if (p.files.size() < 1)    throw std::runtime_error("Specify training, testing and crossvalidation data file.");
    if (p.labels.size() == 0)  throw std::runtime_error("Specify output labels.");

    std::cout << "=== Loading data" << std::endl;
    DataSet train, test, xval;
    train.load_tmp(p.files[0]);
    if (p.files.size() >= 2) test.load_tmp(p.files[1]);
    if (p.files.size() >= 3) xval.load_tmp(p.files[2]);

    Classifier best_one;
    Real best_err = -1.0;

    std::cout << "=== Loading data" << std::endl;

    for (size_t i = 0; i < p.try_count; ++i) {
        std::cout << "--- Classifier #" << (i + 1) << std::endl;
        NeuralNet nn(test.sample(0).first.size(), p.hidden_neurons, test.sample(0).second.size(), sigmoid_func, logsigmoid_func);
        Classifier c;
        Classifier::Teacher t(c, nn, train, (p.files.size() >= 2 ? test : train), ((p.files.size() >= 3) ? xval : train));
        t.teach(p.chunk_size, .5);

        Real err = c.error(test);
        if (best_err < 0.0 || best_err > err) {
            best_err = err;
            best_one = c;
        }
    }

    std::cout << "=== Writing neural net" << std::endl;
    std::ofstream ofs(p.out_file.c_str());
    ofs << best_one;
}

/// classification
void classify(params& p)
{
    if (p.cls_file.empty())    throw std::runtime_error("Specify classifier filename.");
    if (p.files.size() == 0)   throw std::runtime_error("Nothing to classify");

    static const int barsize = 20;

    Classifier c;
    { std::ifstream ifs(p.cls_file.c_str()); ifs >> c; }

    for (size_t i = 0; i < p.files.size(); ++i) {
        DataSet data;
        data.load(p.files[i]);

        Classifier::Vector result = c.exec(data);

        for (size_t l = 0; l < c.labels().size(); ++l) {
            size_t marks = static_cast<int>(20 * std::exp(result(i)) + .5);
            std::cout << std::setw(barsize) << c.labels()[l] << '[' << string(marks, '#') << string(barsize - marks, ' ') << ']' << std::endl;
        }
        std::cout << std::endl;
    }
}

void foo(params& p)
{
    Classifier c;

    {
        NNLayer::Vector vin(2), vout(1);
        DataSet data;
        vin(0) = -1; vin(1) = -1; vout(0) =  1; data.add_sample(vin, vout);
        vin(0) = -1; vin(1) =  0; vout(0) = -1; data.add_sample(vin, vout);
        vin(0) = -1; vin(1) =  1; vout(0) =  1; data.add_sample(vin, vout);
        vin(0) =  0; vin(1) = -1; vout(0) = -1; data.add_sample(vin, vout);
        vin(0) =  0; vin(1) =  0; vout(0) = -1; data.add_sample(vin, vout);
        vin(0) =  0; vin(1) =  1; vout(0) = -1; data.add_sample(vin, vout);
        vin(0) =  1; vin(1) = -1; vout(0) =  1; data.add_sample(vin, vout);
        vin(0) =  1; vin(1) =  0; vout(0) = -1; data.add_sample(vin, vout);
        vin(0) =  1; vin(1) =  1; vout(0) =  1; data.add_sample(vin, vout);
        data.normalize_all();
        //data.write_tmp(std::cout);
        std::cout << std::endl;

        NeuralNet nn(data.sample(0).first.size(), 4, data.sample(0).second.size());
        Classifier::Teacher t(c, nn, data, data, data);
        t.teach(0, .02);
    }

    Classifier::Vector v;
    while (std::cin >> v) std::cout << c.exec(v) << std::endl;
}

/// parse the commandline
void parse_params(int argc, char** argv, params& p)
{
    p.prog = argv[0];
    p.verbose = false;
    p.hidden_neurons = 0;
    p.chunk_size = 20;
    p.try_count = 5;

    if (argc < 2) throw std::runtime_error("Mode needs to be specified, try: " + p.prog + " help");

    string str = argv[1];
         if (str == "help")     p.mode = prog_help;
    else if (str == "foo")      p.mode = foo;
    else if (str == "dataset")  p.mode = do_dataset;
    else if (str == "train")    p.mode = training;
    else if (str == "classify") p.mode = classify;
    else throw std::runtime_error("Unknown mode: " + str);

    for (int i = 2; i < argc; ++i) {
        str = argv[i];
             if (str == "-v") p.verbose = true;
        else if (str == "-o") p.out_file = argv[++i];
        else if (str == "-f") p.cls_file = argv[++i];
        else if (str == "-d") p.data_dir = argv[++i];
        else if (str == "-t") p.try_count      = boost::lexical_cast<size_t>(argv[++i]);
        else if (str == "-h") p.hidden_neurons = boost::lexical_cast<size_t>(argv[++i]);
        else if (str == "-c") p.chunk_size     = boost::lexical_cast<size_t>(argv[++i]);
        else if (str == "-l") boost::algorithm::split(p.labels, argv[++i], boost::algorithm::is_any_of(":"));
        else if (str.substr(0, 1) == "-") throw std::runtime_error("Unrecognized commandline option: " + str);
        else p.files.push_back(str);
    }
}

int main(int argc, char** argv)
{

    try {
        params p;
        parse_params(argc, argv, p);
        p.mode(p);
    } catch (std::runtime_error& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

