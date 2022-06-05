#ifndef BLAS_WRAPPER_H
#define BLAS_WRAPPER_H

// Simple placeholder code for BLAS calls - replace with calls to a real BLAS library
#include "tbb/parallel_reduce.h"
#include "tbb/parallel_for.h"
#include <vector>

namespace BLAS{

// dot products ==============================================================
template<typename T>
inline double dot(const std::vector<T> &x, const std::vector<T> &y)
{ 
    size_t n = x.size();
   //return cblas_ddot((int)x.size(), &x[0], 1, &y[0], 1); 
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n),
        0.f,
        [&](const tbb::blocked_range<size_t>& r, float init) {
            for (size_t a = r.begin(); a != r.end(); ++a) {
                init += x[a]*y[a];
            }
            return init;
        },
        [](float x, float y) {
            return x + y;
        }
        );
   /*double sum = 0;
   for(int i = 0; i < x.size(); ++i)
      sum += x[i]*y[i];
   return sum;*/
}

// inf-norm (maximum absolute value: index of max returned) ==================

template <typename T>
inline int index_abs_max(const std::vector<T> &x)
{ 
    size_t n = x.size();
   //return cblas_idamax((int)x.size(), &x[0], 1); 
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n),
        0,
        [&](const tbb::blocked_range<size_t>& r, size_t max_idx) {
            for (size_t a = r.begin(); a != r.end(); ++a) {
                if (std::abs(x[a]) > std::abs(x[max_idx])) {
                    max_idx = a;
                }
            }
            return max_idx;
        },
        [&](size_t idx0, size_t idx1) {
            if (std::abs(x[idx0]) > std::abs(x[idx1])) {
                return idx0;
            }
            else {
                return idx1;
            }
        }
        );
   /*int maxind = 0;
   double maxvalue = 0;
   for(int i = 0; i < x.size(); ++i) {
      if(fabs(x[i]) > maxvalue) {
         maxvalue = fabs(x[i]);
         maxind = i;
      }
   }
   return maxind;*/
}

// inf-norm (maximum absolute value) =========================================
// technically not part of BLAS, but useful

template <typename T>
inline double abs_max(const std::vector<T> &x)
{ return std::fabs(x[index_abs_max(x)]); }

// saxpy (y=alpha*x+y) =======================================================
template <typename T>
inline void add_scaled(T alpha, const std::vector<T> &x, std::vector<T> &y)
{ 
   //cblas_daxpy((int)x.size(), alpha, &x[0], 1, &y[0], 1); 
    size_t n = x.size();
    tbb::parallel_for((size_t)0, n, [&](size_t i) {
        y[i] += alpha * x[i];
        });
   /*for(int i = 0; i < x.size(); ++i)
      y[i] += alpha*x[i];*/
}

}
#endif
