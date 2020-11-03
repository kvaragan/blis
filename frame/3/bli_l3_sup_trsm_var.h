/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

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


//
// Prototype object-based interfaces.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC0(opname) \
     ( \
       trans_t trans, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     );

GENPROT( trsmsup_ll_ref_var2 )
GENPROT( trsmsup_lu_ref_var2 )


//
// Prototype BLAS-like interfaces with void pointer operands.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       diag_t           diaga, \
       dim_t            m, \
       dim_t            n, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     );

INSERT_GENTPROT_BASIC0( trsmsup_ll_ref_var2 )
INSERT_GENTPROT_BASIC0( trsmsup_lu_ref_var2 )


//
// Static inlined function definitions.
//

BLIS_INLINE dim_t bli_trsmsup_ll_diag_mc
     (
       const dim_t  ii,
       const dim_t  kc_cur,
       const doff_t d0,
       const dim_t  mc,
       const dim_t  mr,
       const dim_t  kc
     )
{

	if ( kc_cur <= ii )
	{
		// Sanity check.
		printf( "bli_trsmsup_ll_diag_mc(): kc_cur <= ii!\n" );
		bli_abort();

		// We are beyond the diagonal block, and therefore we should use the
		// normal value of mc, or whatever remains of the panel.
		//const dim_t ii_left = ii_end - ii;

		//return ( mc <= ii_left ? mc : ii_left )
	}

	/*
	   We want to find a suitable value to use to partition down the current
	   (rank-kc) panel of A that will result in the referenced region of A
	   containing an (approximately) equal number of elements as an mc x kc
	   dense block.

	       desired diagonal block          typical non-diagonal block        
	       partitioning                    partitioning                      
	                                                                         
	        __________ d0                   ______________________________   
	       |           .                   |                              |  
	       |             .                 |                              |  
	       |               .            mc |                              |  
	     x |                 .             |                              |  
	       |                   .           |______________________________|  
	       |                     .                        kc                 
	       |                       .                                         
	       |_________________________.                                       
	                                                                         
	                                                                         
	   If x is the ideal value of mc that we are seeking, the "area" of the
	   lower trapezoidal matrix region of A that we wish to capture, given the
	   current location of the diagonal offset d0, may be approximated as:

	     0.5*x*x + d0*x = mc*kc
	     => 0.5*x*x + d0*x - mc*kc = 0

	   Given a quadratic equation a*x*x + b*x + c = 0, the solution(s) for
	   x may be found via the quadratic formula:

	           -b +/- sqrt( b*b - 4*a*c ) 
	     x  =  --------------------------
	                      2*a
	   In our case,

	     a = 0.5
	     b = d0
	     c = -mc*kc

	   And therefore,

	           -d0 +/- sqrt( d0*d0 - 4*0.5*(-mc*kc) ) 
	     x  =  --------------------------------------
	                           2*0.5

	     x  =  -d0 +/- sqrt( d0*d0 + 2*mc*kc ) 
	*/

	// This function assumes that ii < kc_cur, and therefore we have not
	// yet completed the current diagonal block. First, let's compute mc_bal
	// (balanced mc) using the formula derived above.

	// Notice that we use kc below instead of kc_cur; this instance of kc
	// comes from the expression mc*kc, which captures the typical number of
	// elements in a block of matrix A.

	const dim_t  r      = d0 * d0 + 2 * mc * kc;
	const double sqrt_r = sqrt( ( double )r );
	const dim_t  x      = -d0 + ( dim_t )sqrt_r;

	// Now align x to the next multiple of mr. This is our balanced mc
	// candidate. But we still might not use it...
	const dim_t  mc_bal = ( ( x + mr - 1 ) / mr ) * mr;

	// Calculate the remaining portion of the diagonal block we have yet to
	// compute with.
	const dim_t  k_rem  = kc_cur - ii;

	if ( k_rem <= mc_bal )
	{
		// If mc_bal equals or exceeds the remaining portion of the diagonal
		// block k_rem, use k_rem instead since that's all we have left.
		return k_rem;
	}
	else // if ( mc_bal < k_rem )
	{
		// If mc_bal falls short of k_rem, consider whether extending mc_bal
		// would capture the rest of the diagonal block. If it would, then
		// use k_rem; otherwise, use mc_bal.

		// Compute the area of the region of A referenced if we were to use
		// mc_bal, and then do the same for k_rem. Note that (a) we factor
		// the expressions to save a couple multiply operations, and (b) the
		// arithmetic is done in integer arithmetic and therefore only an
		// approximation.
		#if 1
		const dim_t area_bal = mc_bal * ( d0 + mc_bal / 2 );
		const dim_t area_rem = k_rem  * ( d0 + k_rem  / 2 );

		// Allow a 25% extension to mc_bal. If this gets us to k_rem (or
		// beyond), then use k_rem. Otherwise, stick with mc_bal.
		if ( area_rem <= ( area_bal * 5 ) / 4 ) return k_rem;
		else                                    return mc_bal;
		#else
		return mc_bal;
		#endif
	}
}

