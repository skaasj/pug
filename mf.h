#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <memory>
#include <random>
#include <cmath>
#include <unordered_map>
#include <cassert>
 
#include <stdio.h>
#include "armadillo"
#include "pug_utils.h"

#include "model_generated.h"

namespace pug {

struct MF 
{
    struct Rating
    {
        uint32_t uidx_;
        uint32_t iidx_;
        float rate_;
        Rating(uint32_t uidx=0, uint32_t iidx=0, float rate=0) : uidx_(uidx), iidx_(iidx), rate_(rate) {}
         
    };
    typedef std::shared_ptr<Rating> RatingP;

    // Class member variables 
    std::vector<RatingP> ratings_;
    std::vector<RatingP> trainset_;
    std::vector<RatingP> testset_;
    uint32_t n_ratings_;
    uint32_t n_train_;
    uint32_t n_test_;
    uint32_t n_users_;
    uint32_t n_items_;
    int n_feat_;

    // Model
    arma::mat P_; // n_feat_ X n_users_ latent matrix
    arma::mat Q_; // n_feat_ X n_items_ latent matrix 

    float mean_rate_ = 0.0;

    // map of real user_id to user_index in the rating matrix
    std::unordered_map<uint32_t,uint32_t> uid_to_uidx_;
    std::unordered_map<uint32_t,uint32_t> iid_to_iidx_;

    MF(void) {}
    
    /*
     * Train
     */
    inline void train(const int n_feat, const float stepsz, const float reg, const float momentum, uint32_t round_len, const size_t maxsec, const size_t maxiter, const int itv_test, const int itv_save)
    {
        n_feat_ = n_feat;
        if (round_len > n_train_) {
            printf("round_len (> n_train_) is truncated to n_train_, %d\n", n_train_);
            round_len = n_train_;
        }

        P_ = arma::randu<arma::mat>(n_feat_, n_users_) * 0.1;
        Q_ = arma::randu<arma::mat>(n_feat_, n_items_) * 0.1;
        arma::mat grad_P_inc(n_feat_, n_users_);
        arma::mat grad_Q_inc(n_feat_, n_items_);
        grad_P_inc.zeros();
        grad_Q_inc.zeros();
        
        // compute mean_rate
        for (auto &rating: trainset_) mean_rate_ += rating->rate_;
        mean_rate_ = mean_rate_/n_train_;

        float time_stamp = 0.0;
        uint32_t pos = 0;        // current position in trainset
        uint32_t iter = 0;
        
        while (iter < maxiter && time_stamp < maxsec) 
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            for (int i=0; i<round_len; i++)      
            {
                auto r = trainset_[pos]; 
                auto error = ((r->rate_ - mean_rate_) - arma::dot(P_.col(r->uidx_), Q_.col(r->iidx_)));
                grad_P_inc.col(r->uidx_) = momentum * grad_P_inc.col(r->uidx_) + stepsz * (error * Q_.col(r->iidx_) - reg * P_.col(r->uidx_));
                grad_Q_inc.col(r->iidx_) = momentum * grad_Q_inc.col(r->iidx_) + stepsz * (error * P_.col(r->uidx_) - reg * Q_.col(r->iidx_));
                P_.col(r->uidx_) = P_.col(r->uidx_) + grad_P_inc.col(r->uidx_);
                Q_.col(r->iidx_) = Q_.col(r->iidx_) + grad_Q_inc.col(r->iidx_);
                pos = (pos+1) % n_train_; // to next rating
            } 
            iter++;
            auto t_end = std::chrono::high_resolution_clock::now();
            time_stamp = time_stamp + std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count()/1000000.0;
            // do test              
            if (iter % itv_test == 0) {
                float cur_rmse = compute_rmse(testset_);
                printf("r%d: rmse: %.4f (%.2fs) \n", iter, cur_rmse, time_stamp);
            }
        }
    }

    inline float compute_rmse(const std::vector<RatingP> &dataset)
    {
        float rmse = 0.0;
        for (auto &r: dataset) {
            rmse += pow((r->rate_ - mean_rate_) - arma::dot(P_.col(r->uidx_), Q_.col(r->iidx_)), 2);
        }
        rmse = sqrt(rmse / n_test_);
        return rmse;
    }


    inline void load_from_file(const std::string& filepath, const std::string& delim)
    {
        std::ifstream f(filepath);
        std::string line;
        size_t pos = 0;
        uint32_t max_uidx = 0; // uidx and iidx start from '0'
        uint32_t max_iidx = 0;
        float rate;

        auto t_start = std::chrono::high_resolution_clock::now();
        while (getline(f, line)) 
        {
            uint32_t uid, iid, cur_uidx, cur_iidx; float rate;
            // read three tokens in order of uid, iid, rate
            pos = line.find(delim);
            uid = (uint32_t)std::stoi(line.substr(0, pos));
            line.erase(0, pos + delim.length());
            pos = line.find(delim);
            iid = (uint32_t)std::stoi(line.substr(0, pos));
            line.erase(0, pos + delim.length());
            pos = line.find(delim);
            rate = std::stof(line.substr(0, pos));
            
            // Check if the uid (iid) is already converted to uidx (iidx)
            auto tmp = uid_to_uidx_.find(uid);
            if (tmp == uid_to_uidx_.end()) {
                cur_uidx = max_uidx++;
                uid_to_uidx_[uid] = cur_uidx;
            } else {
                cur_uidx = tmp->second;
            }
            tmp = iid_to_iidx_.find(iid);
            if (tmp == uid_to_uidx_.end()) {
                cur_iidx = max_iidx++;
                iid_to_iidx_[iid] = cur_iidx;
            } else {
                cur_iidx = tmp->second;
            }
            // Add new rating entry
            ratings_.emplace_back(RatingP(new Rating{cur_uidx, cur_iidx, rate}));
        }
        f.close();

        n_ratings_ = ratings_.size();
        n_users_ = max_uidx;
        n_items_ = max_iidx;
        
        std::shuffle(ratings_.begin(), ratings_.end(), std::default_random_engine(1)); // shuffle
        split_train_test(0.9);

        auto t_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
        printf("Loading data took %.2f sec.\n", elapsed/1000000.);
        printf("n_users: %d, n_items: %d, n_train: %d, n_test: %d\n", n_users_, n_items_, n_train_, n_test_);
    }

    int save(const std::string& file) const 
    {
        flatbuffers::FlatBufferBuilder fbb;
        std::vector<RatingP> trainset = trainset_;
        std::vector<RatingP> testset    = testset_;

        // buffer for trainset 
        std::vector<flatbuffers::Offset<pug::Rating>> rs_train;
        for (auto &rating: trainset) {
            rs_train.push_back(pug::CreateRating(fbb, rating->uidx_, rating->iidx_, rating->rate_));
        } 
        // buffer for testset
        std::vector<flatbuffers::Offset<pug::Rating>> rs_test;
        for (auto &rating: testset) {
            rs_test.push_back(pug::CreateRating(fbb, rating->uidx_, rating->iidx_, rating->rate_));
        }

        // build Dataset
        auto dataset = pug::CreateDataset(fbb, fbb.CreateVector(rs_train.data(), rs_train.size()), fbb.CreateVector(rs_test.data(), rs_test.size()), n_train_, n_test_, n_users_, n_items_); 
        fbb.Finish(dataset);
        
        // write to file
        std::ofstream out(file, std::ofstream::out | std::ofstream::binary);
        out.write((const char *)fbb.GetBufferPointer(), fbb.GetSize());

        return 0;
    }

    int load(const std::string& file) 
    {
        std::ifstream in(file, std::ifstream::binary);
        std::stringstream ss;
        ss << in.rdbuf();
        std::string s = ss.str();

        const pug::Dataset *dataset = pug::GetDataset(s.data());
        size_t sz_trainset = dataset->trainset()->Length();
        size_t sz_testset  = dataset->testset()->Length();

        // load numbers
        n_train_ = dataset->n_train();
        n_test_  = dataset->n_test();
        n_users_ = dataset->n_user();
        n_items_ = dataset->n_item();

        assert(n_train_ == sz_trainset);
        assert(n_test_  == sz_testset);

        // load trainset
        for (int i=0; i<sz_trainset; i++) {
            const auto *rating = dataset->trainset()->Get(i);
            trainset_.emplace_back(RatingP(new Rating{rating->uidx(), rating->iidx(), rating->rate()}));
        }
        // load testset
        for (int i=0; i<sz_testset; i++) {
            const auto *rating = dataset->testset()->Get(i);
            testset_.emplace_back(RatingP(new Rating{rating->uidx(), rating->iidx(), rating->rate()}));
        }

        printf("Data loaded from %s\n", file.c_str());
        
        return 0;
    }


    inline void split_train_test(const float trainset_ratio)
    {
        n_train_ = (uint32_t)(n_ratings_ * trainset_ratio);
        n_test_ = n_ratings_ - n_train_;
        for (int i=0; i < n_ratings_; i++) {
            if (i < n_train_) {
                trainset_.push_back(ratings_[i]);
            } else {
                testset_.push_back(ratings_[i]);
            }
        }
    }

    void lookup(std::vector<RatingP> &ratings) 
    {
        RatingP rating;
        while (1) 
        {
            std::string s;
            std::cout << "\nEnter entry index: ";
            std::cin >> s;
            if (s == "q") break;
            if (pug_utils::is_number(s)) {
                auto idx = stoi(s) - 1; 
                if (idx > ratings.size()){
                    printf("index should be smaller than the ratings size %lu\n", ratings.size());
                    continue;
                }
                rating = ratings[idx];
                printf("%d, %d, %f\n", rating->uidx_, rating->iidx_, rating->rate_);
            } else {
                printf("input is not a valid number\n");
                continue;
            }
        }
    }
};

}; // end of namespace
