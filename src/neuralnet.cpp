/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#include "neuralnet.hpp"
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>


NeuralNet::NeuralNet() {}

NeuralNet::NeuralNet(size_t no_inputs, size_t no_hidden, size_t no_outputs, ActivationFunc& hidden_act, ActivationFunc& output_act)
{
    NNLayer hidden_layer(no_inputs, no_hidden, hidden_act);
    NNLayer output_layer(no_hidden, no_outputs, output_act);
    add_layer(hidden_layer);
    add_layer(output_layer);
}

const NNLayer& NeuralNet::layer(size_t idx) const { return layers.at(idx); }

void NeuralNet::add_layer(NNLayer& layer)
{
    // number of current network outputs must be the same as number of inputs of the new layer
    if (layers.size() != 0 && no_outputs() != layer.no_inputs())
        throw std::runtime_error("Number of outputs of last network layer and number of added layer inputs do not match");
    layers.push_back(NNLayer(1, 1, linear_func));
    layers.back().swap(layer);
}

NeuralNet::Vector NeuralNet::exec(const Vector& input) const
{
    Vector x = input;
    for (LayerArray::const_iterator i = layers.begin(); i != layers.end(); ++i)
        x = i->exec(x);
    return x;
}

size_t NeuralNet::no_inputs() const
{
    if (layers.size() == 0) throw std::runtime_error("Empty network");
    return layers[0].no_inputs();
}

size_t NeuralNet::no_outputs() const
{
    if (layers.size() == 0) throw std::runtime_error("Empty network");
    return layers.back().no_outputs();
}

void NeuralNet::load(std::istream& is)
{
    layers.clear();
    NNLayer l(1, 1, linear_func);
    while (is >> l) add_layer(l);
}

std::istream& operator>>(std::istream& is, NeuralNet& nn) { nn.load(is); return is; }

std::ostream& operator<<(std::ostream& os, const NeuralNet& nn)
{
    for (NeuralNet::LayerArray::const_iterator i = nn.get_layers().begin(); i != nn.get_layers().end(); ++i)
        os << *i << std::endl;
    return os;
}



NeuralNet::Teacher::Teacher(NeuralNet& nn)
{
    teachers.resize(nn.layers.size());
    for (size_t i = 0; i < nn.layers.size(); ++i)
        teachers[i] = (new NNLayer::Teacher(nn.layers[i]));
}

NeuralNet::Teacher::~Teacher()
{
    for (size_t i = 0; i < teachers.size(); ++i) delete teachers[i];
}

void NeuralNet::Teacher::teach(Numeric learning_rate)
{
    for (size_t i = 0; i < teachers.size(); ++i) teachers[i]->teach(learning_rate);
}

void NeuralNet::Teacher::sample(const Vector& input, const Vector& output)
{
    // 1. record forward feed results
    std::vector<Vector> results(teachers.size() + 1);
    results[0] = input;
    for (size_t i = 0; i < teachers.size(); ++i)
        results[i + 1] = teachers[i]->get_layer().exec(results[i]);

    Vector delta = output - results.back();
    for (ssize_t i = teachers.size() - 1; i >= 0; --i) {
        //std::cout << "Delta" << i << ": " << delta << std::endl;
        teachers[i]->sample(results[i], results[i + 1], delta);
        delta = prod(delta, trans(teachers[i]->get_layer().weight_matrix()));
        delta = boost::numeric::ublas::vector_range<Vector>(delta, boost::numeric::ublas::range(1, teachers[i]->get_layer().no_inputs() + 1));
    }
}

