/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */



#pragma once
#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "common.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include <map>
#include <string>

class NNLayer;
class NNLayerTeacher;

/**
 * Class coupling implementation of activation function and its derivative.
 */
class ActivationFunc {
    public:
        /// underlying numeric type
        typedef Real NumType;
        /// underlying vector type
        typedef boost::numeric::ublas::vector<NumType> VecType;
        /// scalar vector type
        typedef boost::numeric::ublas::scalar_vector<NumType> ScalarType;

        /**
         * compute activation function
         * @param v potential vector
         */
        virtual VecType  f(const VecType& v) const = 0;

        /**
         * derivative of activation function
         * @param in input vector
         * @param nn neural net layer
         * @param out output vector
         */
        virtual VecType df(const VecType& in, const NNLayer& nn, const VecType& out) const = 0;

        /// activation function name
        virtual std::string name() const = 0;
};

/**
 * Linear activation function
 */
class LinearFunc : public ActivationFunc {
    public:
        virtual VecType  f(const VecType& v) const;
        virtual VecType df(const VecType& in, const NNLayer& nn, const VecType& out) const;
        virtual std::string name() const;
};

/**
 * Sigmoid activation function
 */
class SigmoidFunc : public ActivationFunc {
    public:
        virtual VecType  f(const VecType& v) const;
        virtual VecType df(const VecType& in, const NNLayer& nn, const VecType& out) const;
        virtual std::string name() const;
};

/**
 * Logarithmic sigmoid activation function
 */
class LogSigmoidFunc : public ActivationFunc {
    public:
        virtual VecType  f(const VecType& v) const;
        virtual VecType df(const VecType& in, const NNLayer& nn, const VecType& out) const;
        virtual std::string name() const;
};

/// linear activation function instance
extern LinearFunc linear_func;
/// sigmoid activation function instance
extern SigmoidFunc sigmoid_func;
/// logarithmic sigmoid activation function instance
extern LogSigmoidFunc logsigmoid_func;

/**
 * Neural Network Layer
 */
class NNLayer {
    public:
        typedef Real Numeric; ///< underlying numeric type
        typedef boost::numeric::ublas::matrix<Numeric> Matrix;        ///< matrix type
        typedef boost::numeric::ublas::vector<Numeric> Vector;        ///< vector type
        typedef boost::numeric::ublas::scalar_vector<Numeric> Scalar; ///< scalar type
        typedef std::map<std::string, const ActivationFunc*> ActFuncMap;

    public:

        /**
         * Neural network layer constructor
         * @param inputs number of inputs
         * @param outputs number of outputs
         * @param a activation function
         */
        NNLayer(size_t inputs, size_t outputs, ActivationFunc& a);

        /**
         * Compute potential of neurons in the layer
         * @param in input vector
         */
        Vector potential(const Vector& in) const;

        /**
         * Compute layer output
         * @param in input vector
         */
        Vector exec(const Vector& in) const;

        /**
         * randomize weight vectors
         * @param lo lower bound
         * @param hi upper bound
         */
        void randomize(Real lo = -1.0, Real hi = +1.0);

        /// get weight matrix
        const Matrix& weight_matrix() const { return weights; }
        /// get activation function
        const ActivationFunc& activation() const { return *act; }
        /// get input vector size
        size_t no_inputs() const { return weights.size1() - 1; }
        /// get output vector size
        size_t no_outputs() const { return weights.size2(); }

        /// load from input stream
        void load(std::istream& is);

        /// swap two layer contents
        void swap(NNLayer& l);

    public:
        /**
         * Layer learning context.
         * Can be used for both, sample-by-sample and burst learning.
         */
        class Teacher {
            public:
                /// Create a learning context for a layer
                explicit Teacher(NNLayer& nn, bool randomize = true);
                /// present a training dataset sample and difference from desired value, thus updating @a dw
                void sample(const Vector& in, const Vector& out, const Vector& dout);
                /// present a training dataset sample and difference from desired value, thus updating @a dw
                void sample(const Vector& in, const Vector& dout);
                /// update layer weights wrt. accumulated weight deltas from samples and given learning rate
                void teach(Numeric learning_rate);
                // get layer
                const NNLayer& get_layer() const { return *layer; }
            private:
                Matrix dw;      ///< accumulated weight deltas
                NNLayer* layer; ///< layer reference
        };

        typedef std::auto_ptr<Teacher> TeacherPtr;
        friend class Teacher;

    public:
        /// register activation function for the factory
        static void register_activation(const ActivationFunc& f);

    private:
        const ActivationFunc* act; ///< activation function 
        Matrix weights;            ///< weight matrix
    private:
        static ActFuncMap act_map; ///< activation function factory map
        static Vector input_vec(const Vector& in); ///< prepend zero-th input element (value 1.0)
};

/// Layer input
std::istream& operator>>(std::istream& is, NNLayer& nn);
/// Layer output
std::ostream& operator<<(std::ostream& os, const NNLayer& nn);


#endif

