/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#include "classifier.hpp"
#include <iostream>
#include <iomanip>
#include <boost/numeric/ublas/io.hpp>

struct params {
    void(*mode)(params&);
    std::string prog;
};

void prog_help(params& p)
{
    std::cout << p.prog << " <mode> <switches>" << std::endl
              << " mode is one of: train, classify, dataset, help" << std::endl
              << "" << std::endl
              << "" << std::endl;
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

        NeuralNet nn(2, 4, 1);
        Classifier::Teacher t(c, nn, data, data, data);
        t.teach(0, .02);
    }

    Classifier::Vector v;
    while (std::cin >> v) std::cout << c.exec(v) << std::endl;
}

void parse_params(int argc, char** argv, params& p)
{
    p.prog = argv[0];
    if (argc < 2) throw std::runtime_error("Mode needs to be specified, try: " + p.prog + " help");

    std::string str = argv[1];
         if (str == "help") p.mode = prog_help;
    else if (str == "foo")  p.mode = foo;
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


/*
    if (argc == 2) {
        LabelList l; l.push_back("rock"); l.push_back("pop");
        DataSet data;
        std::cout << "load" << std::endl;
        data.load_dir(argv[1], l);
        std::cout << "normalize" << std::endl;
        data.normalize_all();
        std::cout << "shuffle" << std::endl;
        data.shuffle();
        std::cout << "write" << std::endl;
        data.write_tmp("data.txt");
        return 0;
    }



    NeuralNet::Teacher t(nn);
    NeuralNet nn2 = nn;
    std::cout << nn << std::endl;
    std::cout << nn << std::endl;
    while (std::cin >> v)
        std::cout << nn2.exec(v) << " " << nn.exec(v) << std::endl;
    if (argc != 2) return 1;
    Features f(argv[1]);
    for (size_t n = 0; n < f.frames(); ++n) {
        const FeatureVector& v = f.feature(n);
        std::cout << "Frame " << std::setw(4) << n << ": " << v;
        std::cout << std::endl;
    }
*/
    return 0;
}

