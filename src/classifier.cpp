/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */

#include "classifier.hpp"
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

using namespace boost::numeric::ublas;

const char LABEL_DELIM[] = ";";

Classifier::Classifier() {}

Classifier::Vector Classifier::exec(const Vector& in) const
{
    return nn.exec(element_div(in - mean(), stddev()));
}

Classifier::Vector Classifier::exec(const DataSet& data) const
{
    Vector sum = zero_vector<Real>(labels_.size());
    for (size_t i = 0; i < data.count(); ++i)
        sum += exec(data.sample(i).first);
    // take avarege of the results
    return element_div(sum, scalar_vector<Real>(sum.size(), data.count()));
}

Real Classifier::error(const Vector& in, const Vector& out) const
{
    Vector err = out - exec(in);
    return sum(element_prod(err, err)); // err^2
}

Real Classifier::error(const DataSet& data) const
{
    Real err = 0.0;
    for (size_t i = 0; i < data.count(); ++i)
        err += error(data.sample(i).first, data.sample(i).second);
    return err;
}

void Classifier::load(std::istream& is)
{
    std::string str;
    getline(is, str);
    boost::algorithm::split(labels_, str, boost::algorithm::is_any_of(LABEL_DELIM));
    is >> mean_ >> stddev_ >> nn;
}

const Classifier::Vector&    Classifier::mean()       const { return mean_; }
const Classifier::Vector&    Classifier::stddev()     const { return stddev_; }
const             NeuralNet& Classifier::neural_net() const { return nn; }
const             LabelList& Classifier::labels()     const { return labels_; }

std::istream& operator>>(std::istream& is, Classifier& c) { c.load(is); return is; }

std::ostream& operator<<(std::ostream& os, const Classifier& c)
{
    for (size_t i = 0; i < c.labels().size(); ++i) { if (i) os << LABEL_DELIM; os << c.labels()[i]; }
    return os << std::endl << c.mean() << ' ' << c.stddev() << std::endl << c.neural_net();
}


Classifier::Teacher::Teacher(Classifier& c, DataSetRef train, DataSetRef test, DataSetRef xval) :
    cls(c), net(new NeuralNet::Teacher(c.nn)), train(train), test(test), xval(xval) {}

void Classifier::Teacher::present(size_t n, size_t offset, Real learning_rate)
{
    size_t end = std::min(offset + n, train.count());
    for (size_t i = offset; i < end; ++i)
        net->sample(train.sample(i).first, train.sample(i).second);
    net->teach(learning_rate);
}

void Classifier::Teacher::present(size_t n, Real learning_rate)
{
    for (size_t i = 0; i < train.count(); i += n) present(n, i, learning_rate);
}

void Classifier::Teacher::teach(size_t n, Real init_learning_rate)
{
    assert(false);
}

Real Classifier::Teacher::train_error() const { return cls.error(train); }
Real Classifier::Teacher::test_error()  const { return cls.error(test); }
Real Classifier::Teacher::xval_error()  const { return cls.error(xval); }

