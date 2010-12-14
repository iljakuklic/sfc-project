/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */

#pragma once
#ifndef NEURALNET_HPP_
#define NEURALNET_HPP_

#include "layer.hpp"


/**
 * Backpropagation neural network
 */
class NeuralNet {
    public:
        typedef std::vector<NNLayer> LayerArray;
        typedef NNLayer::Numeric Numeric;
        typedef NNLayer::Vector Vector;
        typedef NNLayer::Matrix Matrix;

    public:
        /// create an empty neural network
        explicit NeuralNet();
        /// create an  neural network with one hidden layer
        explicit NeuralNet(size_t no_inputs, size_t no_hidden, size_t no_outputs, ActivationFunc& hidden_act = sigmoid_func, ActivationFunc& output_act = linear_func);
        /// return layers array
        const LayerArray& get_layers() const { return layers; }
        /// get an layer
        const NNLayer& layer(size_t idx) const;
        /// add layer to the end of the network.
        /// (destructively, net takes layer ownership, original layer gets cleared)
        void add_layer(NNLayer& layer);
        /// compute network output
        Vector exec(const Vector& input) const;
        /// get input vector size
        size_t no_inputs() const;
        /// get output vector size
        size_t no_outputs() const;
        /// load from input stream
        void load(std::istream& is);

    public:

        /// NeuralNet learning context
        class Teacher {
            public:
                typedef std::vector<NNLayer::Teacher*> LayerTeacherArray;

                /// New network learning context
                explicit Teacher(NeuralNet& nn, bool randomize = true);
                /// Destructor
                ~Teacher();
                /// present sample vector and its desired output
                void sample(const Vector& input, const Vector& output);
                /// update weight vectors
                void teach(Numeric learning_rate);
            private:
                /// individual layer teachers
                LayerTeacherArray teachers;
        };

        friend class Teacher;

    private:
        LayerArray layers;
};

std::istream& operator>>(std::istream& is, NeuralNet& nn);
std::ostream& operator<<(std::ostream& os, const NeuralNet& nn);

#endif

