/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#include "layer.hpp"

#include <cmath>
#include <ctime>
#include <cstdlib>

#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

using namespace boost::numeric::ublas;

LinearFunc::VecType LinearFunc:: f(const VecType& v) const { return v; }
LinearFunc::VecType LinearFunc::df(const VecType& in, const NNLayer& nn, const VecType& out) const { return ScalarType(out.size(), 1.0); }
std::string LinearFunc::name() const { return "linear"; }

SigmoidFunc::VecType SigmoidFunc:: f(const VecType& v) const { VecType r(v.size()); std::transform(v.begin(), v.end(), r.begin(), sigmoid); return r; }
SigmoidFunc::VecType SigmoidFunc::df(const VecType& in, const NNLayer& nn, const VecType& out) const { return element_prod(ScalarType(out.size(), 1.0) - out, out); }
std::string SigmoidFunc::name() const { return "sigmoid"; }

SigmoidFunc::VecType LogSigmoidFunc:: f(const VecType& v) const { VecType r(v.size()); std::transform(v.begin(), v.end(), r.begin(), logsigmoid); return r; }
SigmoidFunc::VecType LogSigmoidFunc::df(const VecType& in, const NNLayer& nn, const VecType& v) const
    { VecType r(v.size()); std::transform(v.begin(), v.end(), r.begin(), static_cast<Real(*)(Real)>(std::exp)); return ScalarType(r.size(), 1.0) - r; }
std::string LogSigmoidFunc::name() const { return "logsigmoid"; }

LinearFunc linear_func;
SigmoidFunc sigmoid_func;
LogSigmoidFunc logsigmoid_func;

NNLayer::NNLayer(size_t inputs, size_t outputs, ActivationFunc& a) : act(&a), weights(inputs + 1, outputs)
{
    // lazy act_map initialisation with default act. funcs
    if (act_map.size() == 0) {
        register_activation(linear_func);
        register_activation(sigmoid_func);
        register_activation(logsigmoid_func);
    }
}

NNLayer::Vector NNLayer::potential(const Vector& in) const
{
    assert(in.size() + 1 == weights.size1());
    return prod(input_vec(in), weights);
}

NNLayer::Vector NNLayer::exec(const Vector& in) const { return act->f(potential(in)); }

void NNLayer::randomize(Real lo, Real hi)
{
    static boost::mt19937 rng(time(0));
    boost::uniform_real<Real> dist(lo, hi);
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random(rng, dist);

    for (size_t i1 = 0; i1 < weights.size1(); ++i1)
        for (size_t i2 = 0; i2 < weights.size2(); ++i2)
            weights(i1, i2) = random();
}

void NNLayer::load(std::istream& is)
{
    std::string actname;
    is >> actname >> weights;
    act = act_map[actname];
    //if (!act) throw std::runtime_error("Unknown activation function: '" + actname + "'");
}

void NNLayer::swap(NNLayer& l)
{
    std::swap(act, l.act);
    l.weights.swap(weights);
}

std::istream& operator>>(std::istream& is, NNLayer& nn) { nn.load(is); return is; }
std::ostream& operator<<(std::ostream& os, const NNLayer& nn) { return os << nn.activation().name() << " " << nn.weight_matrix(); }

void NNLayer::register_activation(const ActivationFunc& f) { act_map[f.name()] = &f; }
NNLayer::ActFuncMap NNLayer::act_map = NNLayer::ActFuncMap();

NNLayer::Vector NNLayer::input_vec(const Vector& in)
{
    Vector input(in.size() + 1);
    input(0) = 1.0;
    noalias(vector_range<Vector>(input, range(1, in.size() + 1))) = in;
    return input;
}

NNLayer::Teacher::Teacher(NNLayer& nn, bool randomize)
    : n_data(0), dw(zero_matrix<Numeric>(nn.weights.size1(), nn.weights.size2())), layer(&nn)
{
    if (randomize) layer->randomize();
    w2 = nn.weights;
}

void NNLayer::Teacher::sample(const Vector& in, const Vector& out, const Vector& dout)
{
    assert(in.size() + 1 == dw.size1());
    assert(dout.size() == dw.size2());
    assert(dout.size() == out.size());
    // compute weight deltas and add them to the dw matrix
    dw += outer_prod(input_vec(in), element_prod(layer->activation().df(in, *layer, out), dout));
    ++n_data;
}

void NNLayer::Teacher::teach(Numeric learning_rate)
{
    // momentum constant
    static const Real momentum = 0.1;
    // update weights by dw * learning_rate + momnetum * (weight - prev_weight)
    Matrix update = element_prod(scalar_matrix<Numeric>(dw.size1(), dw.size2(), learning_rate / n_data), dw)
                  + element_prod(scalar_matrix<Numeric>(dw.size1(), dw.size2(), momentum), layer->weights - w2);
    w2 = layer->weights; // store previous weights
    layer->weights += update;
    // zero dw matrix
    dw = scalar_matrix<Numeric>(dw.size1(), dw.size2(), 0.0);
    n_data = 0;
}



