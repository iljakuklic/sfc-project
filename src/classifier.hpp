/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */

#pragma once
#ifndef CLASSIFIER_HPP_
#define CLASSIFIER_HPP_

#include "neuralnet.hpp"
#include "features.hpp"

/**
 * Classifier capable of merging several results together.
 * One sound record usually yields several feature vectors.
 * This classifier blends them together into a single result.
 */
class Classifier {
    public:
        typedef NNLayer::Numeric Numeric;
        typedef NNLayer::Vector Vector;

    public:

        /// default constructor
        explicit Classifier();
        
        /// classify single data frame
        Vector exec(const Vector& in) const;
        /// classify dataset
        Vector exec(const DataSet& data) const;

        /// classification error of an data frame
        Real error(const Vector& in, const Vector& out) const;
        /// classification error of an dataset
        Real error(const DataSet& data) const;

        /// load from input stream
        void load(std::istream& is);

        /// get mean
        const Vector& mean() const;
        /// get standard deviation
        const Vector& stddev() const;
        /// get neural network
        const NeuralNet& neural_net() const;
        /// get output labels
        const LabelList& labels() const;

    public:
        
        /// Classifier learning context
        class Teacher {
            public:
                typedef std::auto_ptr<NeuralNet::Teacher> NetTeacherPtr;
                typedef const DataSet& DataSetRef;

            public:

                /**
                 * constructor
                 * @param c Classifier to train
                 * @param train training data set
                 * @param test testing data set
                 * @param xval crossvalidation data set
                 */
                explicit Teacher(Classifier& c, const NeuralNet& network, DataSetRef train, DataSetRef test, DataSetRef xval);

                /**
                 * present several samples and propagate through the network
                 * @param n number of samples to present
                 * @param offset the index of the first sample form trainig set to present
                 * @param learning_rate neural network learning rate
                 */
                void present(size_t n, size_t offset, Real learning_rate);

                /**
                 * train classifier with training dataset split into chunks by n samples
                 * @param n number of samples to present at a time, whole training set if 0
                 * @param learning_rate neural network learning rate
                 */
                void present(size_t n, Real learning_rate);

                /**
                 * train classifier with training dataset split into chunks by n samples,
                 * dynamically adjusting learning rate and stopping when error stops falling
                 * (implements New Bob algorithm)
                 * @param n number of samples to present at a time, whole training set if 0
                 * @param init_learning_rate initial neural network learning rate
                 */
                bool teach(size_t n, Real init_learning_rate);

                /// calculate error on training data
                Real train_error() const;
                /// calculate error on test data
                Real test_error() const;
                /// calculate error on crossvalidation data
                Real xval_error() const;

            private:
                Classifier& cls;
                NetTeacherPtr net;
                DataSetRef train, test, xval;
        };

        friend class Teacher;

    private:
        NeuralNet nn;          ///< neural network
        Vector mean_, stddev_; ///< normalization constants
        LabelList labels_;     ///< output labels
};

// IO
std::istream& operator>>(std::istream& is, Classifier& c);
std::ostream& operator<<(std::ostream& os, const Classifier& c);

#endif // CLASSIFIER_HPP_

