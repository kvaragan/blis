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

#include "blis.h"

err_t bli_gemmsup_ref
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	// This function implements the default gemmsup handler. If you are a
	// BLIS developer and wish to use a different gemmsup handler, please
	// register a different function pointer in the context in your
	// sub-configuration's bli_cntx_init_*() function.

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

#if 0
	// NOTE: This special case handling is done within the variants.

	// If alpha is zero, scale by beta and return.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// If A or B has a zero dimension, scale C by beta and return early.
	if ( bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return BLIS_SUCCESS;
	}
#endif

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop.
	bli_rntm_set_ways_from_rntm_sup
	(
	  bli_obj_length( c ),
	  bli_obj_width( c ),
	  bli_obj_width( a ),
	  rntm
	);

#if 0
	printf( "rntm.pack_a = %d\n", ( int )bli_rntm_pack_a( rntm ) );
	printf( "rntm.pack_b = %d\n", ( int )bli_rntm_pack_b( rntm ) );

	//bli_rntm_set_pack_a( 0, rntm );
	//bli_rntm_set_pack_b( 0, rntm );
#endif

	return
	bli_l3_sup_thread_decorator
	(
	  bli_gemmsup_int,
	  BLIS_GEMM, // operation family id
	  alpha,
	  a,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm
	);
}

// -----------------------------------------------------------------------------

err_t bli_trsmsup_ref
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	// This function implements the default trsmsup handler. If you are a
	// BLIS developer and wish to use a different trsmsup handler, please
	// register a different function pointer in the context in your
	// sub-configuration's bli_cntx_init_*() function.

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b, &BLIS_ZERO, b, cntx );

#if 0
	// NOTE: This special case handling is done within the variants.

	// If B has a zero dimension, there is nothing to do; return early.
	if ( bli_obj_has_zero_dim( b ) )
		return BLIS_SUCCESS;
#endif

	// Parse and interpret the contents of the rntm_t object to properly set
	// the ways of parallelism for each loop, and then make any additional
	// modifications necessary for the current operation. Notice that we choose
	// the m, n, and k dimension parameters under the assumption that the
	// operation will be a left side case since all right side cases will be
	// transformed to left side cases within bli_trsmsup_int().
	bli_rntm_set_ways_for_trsmsup
	(
	  side,
	  bli_obj_length( b ),
	  bli_obj_width( b ),
	  bli_obj_width( a ),
	  rntm
	);

#if 0
	printf( "rntm.pack_a = %d\n", ( int )bli_rntm_pack_a( rntm ) );
	printf( "rntm.pack_b = %d\n", ( int )bli_rntm_pack_b( rntm ) );

	//bli_rntm_set_pack_a( 0, rntm );
	//bli_rntm_set_pack_b( 0, rntm );
#endif

	// Since the thread decorator doesn't take side arguments, we have to
	// communicate that information by passing the triangular matrix A as
	// the second "B" argument if side is BLIS_RIGHT. Also, since the
	// thread decorator uses a standard gemm suite of arguments, we have
	// to pass in dummy objects for "beta" and "C".
	if ( bli_is_left( side ) )
	{
		return
		bli_l3_sup_thread_decorator
		(
		  bli_trsmsup_int,
		  BLIS_TRSM, // operation family id
		  alpha,
		  a,
		  b,
		  &BLIS_ZERO,
		  b,
		  cntx,
		  rntm
		);
	}
	else // if ( bli_is_right( side ) )
	{
		return
		bli_l3_sup_thread_decorator
		(
		  bli_trsmsup_int,
		  BLIS_TRSM, // operation family id
		  alpha,
		  b,
		  a,
		  &BLIS_ZERO,
		  b,
		  cntx,
		  rntm
		);
	}
}

