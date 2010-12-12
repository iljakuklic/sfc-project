/*
 * SFC project (2010) - music genre classifier
 * by Lukas Kuklinek <xkukli01@stud.fit.vutbr.cz>
 * Faculty of Information Tachnology
 * Brno University of Technology
 */


#pragma once
#ifndef FEATURES_HPP_
#define FEATURES_HPP_

#include "common.hpp"
#include <memory>
#include <map>

namespace Aquila {
    /// forward declaration
    class MfccExtractor;
}

/// type for data annotations
typedef std::map<std::string, Real> AnnotationType;

/**
 * Class representing a sequence of feature vectors for a vaw record.
 */
class Features {
    public:
        /// extract features from a .wav file
        explicit Features(const std::string& filename);
        /// destroy features object
        ~Features();
        /// number of post-processed frames
        size_t frames() const;
        /// get post-processed feature vector for given frame
        FeatureVector feature(size_t frame) const;
    protected:
        /// number of raw frames
        size_t raw_frames() const;
        /// get raw feature vector for given frame
        FeatureVector raw_feature(size_t frame) const;
    private:
        std::auto_ptr<Aquila::MfccExtractor> fea;
};

/**
 * Training, testing, or crossvalidation data set.
 */
class DataSet {
    public:
        /// single data sample type (.first = features, .second = output)
        typedef std::pair<FeatureVector, FeatureVector> DataSample;
        /// type of vector of training samples
        typedef std::vector<DataSample> DataSampleList;

    public:

        /// constructor
        explicit DataSet();

        /**
         * Load training data from a file.
         * This function expects <filename_base>.wav to be an audio file
         * and <filename_base>.tag to be a file with annotations (if parse_annotations is true).
         * Data are added to the dataset.
         */
        void load(const std::string& filename_base, const LabelList& labels = LabelList());

        /// Load data from all files in given directory (non-recursively).
        void load_dir(const std::string& dirname, const LabelList& labels);

        /// Load data from temporary format
        void load_tmp(const std::string& filename);

        /// Write data to temporary format
        void write_tmp(const std::string& filename) const;

        /// add a data sample
        void add_sample(const FeatureVector& in, const FeatureVector& out);

        /// Clear dataset.
        void clear();

        /// Randomly shuffle all samples.
        void shuffle();

        /// In-place normalize an input vector
        void normalize(FeatureVector& vec) const;

        /// Normalize the dataset using mean and stddev
        void normalize_all(FeatureVector m = FeatureVector(), FeatureVector s = FeatureVector());
        
        /// get data sample
        const DataSample& sample(size_t i) const { return samples[i]; }
        /// data samples count
        size_t count() const;
        /// data mean
        FeatureVector mean() const;
        /// data standard deviation
        FeatureVector stddev() const;

        /// Load data annotations
        static AnnotationType load_annotations(const std::string& filename);

        /// get label list
        //const LabelList& get_labels() const;

    private:
        //LabelList labels;         ///< label list
        DataSampleList samples;   ///< list of data samples
        FeatureVector sum, sumsq; ///< stats for data normalization
        bool normalized;          ///< normalization status
};

#endif // FEATURES_HPP_

