/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014 - 2020, The University of Texas at Austin

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

void bli_rntm_set_ways_for_trsmsup
     (
       side_t  side,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     )
{
#if 0
printf( "bli_rntm_set_ways_for_trsm()\n" );
bli_rntm_print( rntm );
#endif

	// Set the number of ways for each loop depending on the side parameter.
	if ( bli_is_left( side ) )
	{
		// The order of the dimension parameters was selected assuming a
		// left side case.
		bli_rntm_set_ways_from_rntm_sup( m, n, k, rntm );
	}
	else // if ( bli_is_right( side ) )
	{
		// Right side cases are always transformed to left side cases, which
		// results in a swapping of m and n. This is because the trsmsup
		// implementation does not handle right side cases directly, but
		// rather by transposing the operation into a left side case.
		bli_rntm_set_ways_from_rntm_sup( n, m, k, rntm );
	}

	// Now modify the number of ways because the parallelism we allow in
	// trsm is limited.
	
	// For now, we limit the parallelism in trsm to the jc and jr loops.

	dim_t jc = bli_rntm_jc_ways( rntm );
	dim_t pc = bli_rntm_pc_ways( rntm );
	dim_t ic = bli_rntm_ic_ways( rntm );
	dim_t jr = bli_rntm_jr_ways( rntm );
	dim_t ir = bli_rntm_ir_ways( rntm );

	// Notice that, when updating the ways, we don't need to update the
	// num_threads field since we only reshuffle where the parallelism is
	// extracted, not the total amount of parallelism.

	// Also, we don't need to take the side into account when limiting the
	// parallelism since the side does not affect the order of the loops, and
	// even if it did, we implement trsmsup so that right side cases will be
	// transformed into their equivalent left side cases.
#if 0
	bli_rntm_set_ways_only
	(
	  jc * ic * pc,
	  1,
	  1,
	  jr * ir,
	  1,
	  rntm
	);
#else
	bli_rntm_set_ways_only
	(
	  jc * pc,
	  1,
	  ic,
	  jr * ir,
	  1,
	  rntm
	);
#endif
}

