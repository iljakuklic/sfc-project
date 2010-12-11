/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#include "features.hpp"
#include "neuralnet.hpp"
#include <iostream>
#include <iomanip>
#include <boost/numeric/ublas/io.hpp>

int main(int argc, char** argv)
{
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

    NeuralNet nn(2, 3, 1);
    NNLayer::Vector v, vin(2), vout(1);
    NeuralNet::Teacher t(nn);
    NeuralNet nn2 = nn;
    std::cout << nn << std::endl;
    for (size_t i = 0; i < 1000000; ++i) {
        vin(0) = -1; vin(1) = -1; vout(0) =  1; t.sample(vin, vout);
        vin(0) = -1; vin(1) =  0; vout(0) = -1; t.sample(vin, vout);
        vin(0) = -1; vin(1) =  1; vout(0) =  1; t.sample(vin, vout);
        vin(0) =  0; vin(1) = -1; vout(0) = -1; t.sample(vin, vout);
        vin(0) =  0; vin(1) =  0; vout(0) = -1; t.sample(vin, vout);
        vin(0) =  0; vin(1) =  1; vout(0) = -1; t.sample(vin, vout);
        vin(0) =  1; vin(1) = -1; vout(0) =  1; t.sample(vin, vout);
        vin(0) =  1; vin(1) =  0; vout(0) = -1; t.sample(vin, vout);
        vin(0) =  1; vin(1) =  1; vout(0) =  1; t.sample(vin, vout);
        t.teach(.1);
        if (i % 1000 == 0) std::cout << i << ": " << nn << std::endl;
    }
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
    return 0;
}

