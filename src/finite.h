/* isfinite without the C library's macro. MinGW's isfinite dispatches on
 * sizeof through float casts, and GCC 15's -Wfloat-conversion flags it for a
 * double argument under -O3. fabs(x) <= DBL_MAX is false exactly for the
 * infinities and NaN; a float argument widens to double without loss. */
#ifndef SMOLMOE_FINITE_H
#define SMOLMOE_FINITE_H

#include <float.h>
#include <math.h>

static inline int sm_finite(double x) { return fabs(x) <= DBL_MAX; }

#endif
