/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       dim_t               m, \
       dim_t               n, \
       dim_t               k1x, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a1x, \
       ctype*     restrict a11, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict d11, \
       ctype*     restrict bx1, \
       ctype*     restrict b11, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	/* This is a combined gemmtrsm_ll/gemm kernel, which is used when
       implementing trsm (left side, lower-triangular):

	     b11 = alpha * b11 - a10 * b01;
	     b11 = inv(a11) * b11;
	     c11 = b11;

       It makes the following assumptions:
       - if k10 < k:
         - A is a lower-trapezoidal matrix, consisting of an m x k10 rectangular
	       part a10 and an m x m triangular part a11, and
	     - B is a general matrix, with k10 x n part b01 and m x n part b11;
       - if k10 == k:
         - A is a rectangular matrix consisting entirely of m x k a10, and
         - B is a rectangular matrix consisting entirely of k x n b01;
	   - b01 and b11 are packed (and row-stored);

       Note that:
	   - If k = m, then k10 = 0 and a1x * bx1 is a rank-0 matrix product and thus
	     the gemm component of the operation is effectively skipped (though b11
	     is still scaled by alpha).
       - If k10 == k, then the kernel call consists of only the gemm component
	     since the diagonal does not intersect the current panel of A.
	*/ \
\
	ctype* minus_one = PASTEMAC(ch,m1); \
\
	ctype* o11; \
	inc_t  rs_o; \
	inc_t  cs_o; \
\
	/* o11 is the 'c' input/output matrix for the gemm component of the
	   microkernel call. When performing a gemmtrsm operation, o11 points
	   to b11 since we must first update B before proceeding to update it
	   with the trsm component below (which in turn updates both B and C).
	   But when performing a gemm-only operation, o11 points to c11 since
	   those invocations do not update B and instead update C directly. */ \
	if ( k1x < k ) \
	{ \
		/* gemm component of gemmtrsm subproblem:
		   b11 = alpha * b11 - a10 * b01; */ \
		o11  = b11; \
		rs_o = rs_b; \
		cs_o = cs_b; \
	} \
	else /* if ( k1x == k ) */ \
	{ \
		/* gemm component of gemm-only subproblem:
		   c11 = alpha * c11 - a10 * b01; */ \
		o11  = c11; \
		rs_o = rs_c; \
		cs_o = cs_c; \
	} \
\
\
	/* gemm (lower triangular trsm):
	   b11 = alpha * b11 - a10 * b01; */ \
\
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		/* The output matrix 'c' being updated is actually b11. */ \
		ctype* restrict ci = &o11[ i*rs_o ]; \
		ctype* restrict ai = &a1x[ i*rs_a ]; \
\
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict cij = &ci [ j*cs_o ]; \
			ctype* restrict bj  = &bx1[ j*cs_b ]; \
			ctype           ab; \
\
			PASTEMAC(ch,set0s)( ab ); \
\
			/* Perform a dot product to update the (i,j) element of c. */ \
			for ( dim_t l = 0; l < k1x; ++l ) \
			{ \
				ctype* restrict aij = &ai[ l*cs_a ]; \
				ctype* restrict bij = &bj[ l*rs_b ]; \
\
				PASTEMAC(ch,dots)( *aij, *bij, ab ); \
			} \
\
			/* If alpha is one, subtract ab from c. Otherwise, scale c by alpha
			   and then subtract ab. */ \
			if ( PASTEMAC(ch,eq1)( *alpha ) ) \
			{ \
				PASTEMAC(ch,subs)( ab, *cij ); \
			} \
			else \
			{ \
				PASTEMAC(ch,axpbys)( *minus_one, ab, *alpha, *cij ); \
			} \
		} \
	} \
\
	/* If this is a gemm-only subproblem, the above loops will perform the
	   entirety of the computation and we can return early. */ \
	if ( k1x == k ) return; \
\
	/* trsm_ll: \
	   b11 = inv(a11) * b11;
	   c11 = b11; */ \
\
	for ( dim_t iter = 0; iter < m; ++iter ) \
	{ \
		dim_t i        = iter; \
		dim_t n_behind = i; \
\
		/* alpha11 (or rather, the inverse of alpha11) is stored within the
		   corresponding element of the contiguous vector d. */ \
		ctype* restrict alpha11  = d11 + (i  )*1; \
		ctype* restrict a10t     = a11 + (i  )*rs_a + (0  )*cs_a; \
		ctype* restrict B0       = b11 + (0  )*rs_b + (0  )*cs_b; \
		ctype* restrict b1       = b11 + (i  )*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a10t * B0; */ \
		/* b1 = b1 / alpha11; */ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			ctype* restrict b01     = B0  + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict beta11  = b1  + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict gamma11 = c11 + (i  )*rs_c + (j  )*cs_c; \
			ctype           beta11c = *beta11; \
			ctype           rho11; \
\
			/* beta11 = beta11 - a10t * b01; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( dim_t l = 0; l < n_behind; ++l ) \
			{ \
				ctype* restrict alpha10 = a10t + (l  )*cs_a; \
				ctype* restrict beta01  = b01  + (l  )*rs_b; \
\
				PASTEMAC(ch,axpys)( *alpha10, *beta01, rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */ \
			PASTEMAC(ch,scals)( *alpha11, beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, *gamma11 ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, *beta11 ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC2( gemmtrsmsup_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       dim_t               m, \
       dim_t               n, \
       dim_t               k1x, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a1x, \
       ctype*     restrict a11, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict d11, \
       ctype*     restrict bx1, \
       ctype*     restrict b11, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	bli_abort(); \
}

INSERT_GENTFUNC_BASIC2( gemmtrsmsup_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
