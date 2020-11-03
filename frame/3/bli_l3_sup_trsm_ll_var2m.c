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

#define FUNCPTR_T gemmsup_fp

typedef void (*FUNCPTR_T)
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       diag_t           diaga,
       dim_t            m,
       dim_t            n,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread
     );

//
// -- var2 ---------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY(ftypes_var2,trsmsup_ll_ref_var2);

void bli_trsmsup_ll_ref_var2
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	// NOTE: Regardless of the value of trans, A will is guaranteed to be the
	// triangular matrix and B the general matrix.

	const num_t    dt        = bli_obj_dt( b );

	/*const bool     packa     = bli_rntm_pack_a( rntm );*/
	/*const bool     packb     = bli_rntm_pack_b( rntm );*/
	const bool     packa     = TRUE;
	const bool     packb     = TRUE;

	      uplo_t   uploa;
	const conj_t   conja     = bli_obj_conj_status( a );
	const diag_t   diaga     = bli_obj_diag( a );

	const dim_t    m         = bli_obj_length( b );
	const dim_t    n         = bli_obj_width( b );

	void* restrict buf_a     = bli_obj_buffer_at_off( a );
	      inc_t    rs_a;
	      inc_t    cs_a;

	void* restrict buf_b     = bli_obj_buffer_at_off( b );
	const inc_t    rs_b      = bli_obj_row_stride( b );
	const inc_t    cs_b      = bli_obj_col_stride( b );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var2[dt];


	if ( bli_is_notrans( trans ) )
	{
		// If trans == BLIS_NO_TRANSPOSE, then the original problem was a left
		// side case. No transposition of the operation is needed since the
		// underlying variant expresses trsm as a left side solve. But we still
		// might have to induce a transposition on A if it was passed in with
		// its transposition bit set.

		if ( bli_obj_has_notrans( a ) )
		{
			// If A is encoded with no transposition, we do nothing since the
			// operation is not transposed either.
			uploa = bli_obj_uplo( a );
			rs_a  = bli_obj_row_stride( a );
			cs_a  = bli_obj_col_stride( a );
		}
		else // if ( bli_obj_has_trans( a ) )
		{
			// If A is encoded with transposition, we apply it since the
			// operation is not transposed.
			uploa = bli_obj_uplo_toggled( a );
			rs_a  = bli_obj_col_stride( a );
			cs_a  = bli_obj_row_stride( a );
		}

		// Sanity check: uploa must be BLIS_LOWER. We won't need to pass it into
		// the type-specific variant function call below since it is uplo-specific.
		if ( uploa != BLIS_LOWER ) bli_abort();

		// Invoke the function.
		f
		(
		  packa,
		  packb,
		  conja,
		  diaga,
		  m,
		  n,
		  buf_alpha,
		  buf_a, rs_a, cs_a,
		  buf_b, rs_b, cs_b,
		  cntx,
		  rntm,
		  thread
		);
	}
	else // if ( bli_is_trans( trans ) )
	{
		// If trans == BLIS_TRANSPOSE, then the original problem was a right
		// side case. Transposition of the operation is needed since the
		// underlying variant expresses trsm as a left side solve. However, we
		// know that A is the triangular matrix, so we need only induce a
		// transposition on A and B (that is, we don't need to swap the
		// positions of A and B since that would have already been done by the
		// caller). This amounts to swapping m and n, swapping the strides on B,
		// and possibly swapping the strides and toggling the uplo on A.

		if ( bli_obj_has_notrans( a ) )
		{
			// If A is encoded with no transposition, we still have to apply
			// one since the operation is being transposed.
			uploa = bli_obj_uplo_toggled( a );
			rs_a  = bli_obj_col_stride( a );
			cs_a  = bli_obj_row_stride( a );
		}
		else // if ( bli_obj_has_trans( a ) )
		{
			// If A is encoded with a transposition, it cancels out with the
			// transposition of the operation.
			uploa = bli_obj_uplo( a );
			rs_a  = bli_obj_row_stride( a );
			cs_a  = bli_obj_col_stride( a );
		}

		// Sanity check: uploa must be BLIS_LOWER.
		if ( uploa != BLIS_LOWER ) bli_abort();

		// Invoke the function (transposing the operation).
		f
		(
		  packa,
		  packb,
		  conja,
		  diaga,
		  n,                 // Swap the m and n dimensions.
		  m,
		  buf_alpha,
		  buf_a, rs_a, cs_a, // Strides may have been swapped above depending
		                     // on the transposition bit in A.
		  buf_b, cs_b, rs_b, // Swap the strides of B.
		  cntx,
		  rntm,
		  thread
		);
	}
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
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
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If alpha is zero, set B to zero and return. */ \
	if ( PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			PASTEMAC(ch,setm) \
			( \
			  BLIS_NO_CONJUGATE, \
			  0, \
			  BLIS_NONUNIT_DIAG, \
			  BLIS_DENSE, \
			  m, n, \
			  alpha, \
			  b, rs_b, cs_b \
			); \
		} \
		return; \
	} \
\
	/* Switches for advanced optimizations. */ \
	const bool  quadratic_mc = TRUE; \
	const bool  fuse_packb   = TRUE; \
	const bool  packtria     = TRUE; /* Connect into packa so that it can be */ \
	                                 /* controlled vial rntm_t? */ \
\
	/* Query the context for various blocksizes. */ \
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	/* Nudge KC up to the nearest multiple of MR. This isn't necessary when
	   A is unpacked, but helps ensure that our edge cases are along the bottom
	   and right edges of A for the backwards-iterating trsm_lu. */ \
	const dim_t KC = bli_align_dim_to_mult( KC0, MR ); \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_b; \
	const inc_t jcstep_b = cs_b; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = rs_b; \
	const inc_t icstep_a = rs_a; \
	const inc_t icstep_d = 1; \
\
	const inc_t jrstep_c = cs_b * NR; \
	const inc_t jrstep_b = cs_b * NR; \
\
	const inc_t irstep_c = rs_b * MR; \
	const inc_t irstep_a = rs_a * MR; \
	const inc_t irstep_d = 1    * MR; \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	const l3ukr_t ukr_id = BLIS_GEMMTRSM_L_UKR; \
	PASTECH(ch,gemmtrsmsup_ker_ft) \
               gemmtrsmsup_ker = bli_cntx_get_l3_gemmtrsmsup_ker_dt( dt, ukr_id, cntx ); \
\
	/* Query the context for the packm kernel address and cast it to its
	   function pointer type. NOTE: This function pointer is only used
	   when fusing the packing of B into the JR loop. */ \
	const l1mkr_t ker_id = NR; \
	PASTECH(ch,packm_cxk_ker_ft) \
	                 packb_ker = bli_cntx_get_packm_ker_dt( dt, ker_id, cntx ); \
	const pack_t schema_b = BLIS_PACKED_COL_PANELS; \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = b; \
	ctype* restrict alpha_cast = alpha; \
\
	/* Make local copies of alpha and other scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           alpha_local = *alpha_cast; \
	ctype           one_local   = *PASTEMAC(ch,1); \
	/*ctype           minus_one   = *PASTEMAC(ch,m1);*/ \
\
	auxinfo_t       aux; \
\
	/* Initialize a mem_t entry for A, B, and d. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. An alternative way of initializing the
	   mem_t entries is:

	     bli_mem_clear( &mem_a ); \
	     bli_mem_clear( &mem_b ); \
	     bli_mem_clear( &mem_d ); \
	*/ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
	mem_t mem_d = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */ \
	/*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */ \
	bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
	bszid_t* restrict bszids = bszids_packab; \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
\
	/* Grow the thrinfo_t tree. */ \
	bszid_t*   restrict bszids_jc = bszids; \
	                    thread_jc = thread; \
	bli_thrinfo_sup_grow( BLIS_TRSM, rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end ); \
	const dim_t n_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n_local % NC; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
/*printf( "jj = %d\n", (int)jj );*/ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Grow the thrinfo_t tree. */ \
		bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
		                    thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( BLIS_TRSM, rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t k = m; \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict alpha_use = ( pp == 0 ? &alpha_local : &one_local ); \
\
			ctype* b_use; \
			inc_t  rs_b_use, cs_b_use, ps_b_use; \
			ctype* d_use; \
\
			/* "Dereference" the bszid_t array and thrinfo_t pointers to the
			   previous loop's children. Note that the thrinfo_t node will have
			   already been created by a previous call to bli_thrinfo_grow(),
			   since bszid values of BLIS_NO_PART cause the tree to grow by two
			   (e.g. to the next bszid that is a normal bszid_t value). */ \
			bszid_t* restrict bszids_pb = &bszids_pc[1]; \
			thread_pb = bli_thrinfo_sub_node( thread_pc ); \
\
			/* Determine the packing buffer and related parameters for matrix
			   B. (If B will not be packed, then a_use will be set to point to
			   b and the _b_use strides will be set accordingly.) Then call
			   the packm sup variant chooser. NOTE: We hard-code BLIS_RRR as
			   the stor3_t id because it is only used to differentiate between
			   the RRC and CRC (dot-based) cases and the micropanel cases. By
			   using RRR, we ensure we induce micropanel packing. */ \
			PASTEMAC(ch,packm_sup_b) \
			( \
			  packb, \
			  fuse_packb, \
			  BLIS_BUFFER_FOR_B_PANEL, \
			  BLIS_RRR, \
			  BLIS_NO_TRANSPOSE, \
			  KC,     NC,       /* This "panel of B" is (at most) KC x NC. */ \
			  kc_cur, nc_cur, NR, \
			  &one_local, \
			  b_pc,   rs_b,      cs_b, \
			  &b_use, &rs_b_use, &cs_b_use, \
			                     &ps_b_use, \
			  cntx, \
			  rntm, \
			  &mem_b, \
			  thread_pb  \
			); \
\
			/* Alias b_use so that it's clear this is our current block of
			   matrix B. */ \
			ctype* restrict b_pc_use = b_use; \
\
			/* We don't need to embed the panel stride of B within the auxinfo_t
			   struct because this variant iterates through B in the jr loop,
			   which occurs here, within the macrokernel, not within the
			   microkernel (or millikernel). */ \
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/ \
\
			/* Prune away the zero region above the diagonal of A by modifying
			   the m dimension and the pointers into A and C. */ \
			const inc_t  i0          = pc_start + pp; \
			const dim_t  m_ic        = m - i0; \
			const doff_t diagoffa_pc = 0; \
			ctype* restrict a_pc0    = a_pc + (i0  )*icstep_a; \
			ctype* restrict c_jc0    = c_jc + (i0  )*icstep_c; \
/*
printf( "i0 = %d  m_ic = %d\n", (int)i0, (int)m_ic ); \
*/ \
\
			/* Determine the packing buffer to for the diagonal of A, then pack
			   the diagonal elements that packing buffer, whose address will be
			   stored to d_use. */ \
			PASTEMAC(ch,packm_sup_d) \
			( \
			  diaga, \
			  KC, \
			  kc_cur, \
			  a_pc0, rs_a, cs_a, \
			  &d_use, \
			  cntx, \
			  rntm, \
			  &mem_d, \
			  thread_pb  \
			); \
\
/*PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: d11", 1, kc_cur, d_use, 1, 1, "%5.2f", "" );*/ \
\
			/* Grow the thrinfo_t tree. */ \
			bszid_t* restrict bszids_ic = &bszids_pb[1]; \
			                  thread_ic = bli_thrinfo_sub_node( thread_pb ); \
			bli_thrinfo_sup_grow( BLIS_TRSM, rntm, bszids_ic, thread_ic ); \
\
			/* We use a separate variable for the IC loop bound because it may
			   change when multithreading. */ \
			dim_t m_end = m_ic; \
			dim_t mc_cur; \
\
			/*bli_thread_range_sub( thread_ic, m_ic, MR, FALSE, &ic_start, &ic_end );*/ \
			/*const dim_t m_local = ic_end - ic_start;*/ \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
			/*const dim_t ic_left =   m_local % MC;*/ \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			/*for ( dim_t ii = ic_start; ii < ic_end; ii += MC )*/ \
			/*for ( dim_t ii = ic_start; ii < ic_end; ii += mc_cur )*/ \
			for ( dim_t ii = 0; ii < m_end; ii += mc_cur ) \
			{ \
				/* "Dereference" the bszid_t array to the previous loop's child. */ \
				bszid_t* restrict bszids_pa = &bszids_ic[1]; \
\
				/* Use the same ic, jr, and ir loops to handle both the block-panel
				   trsm operation with the kc x kc lower triangular block of A as well
				   as the rank-kc gemm that transpires afterward. NOTE: This code
				   (specifically, the "ii == kc_cur" conditional) assumes that the
				   mc_cur values are chosen so that the last one that intersects the
				   diagonal is chosen to end at the last diagonal element in the
				   current rank-kc panel. */ \
				if ( ii == 0 ) \
				{ \
					/* "Dereference" the thrinfo_t pointer to the previous loop's
					   sub-prenode, which is a branch of the thrinfo_t tree that
					   contains communicators and threading info for the initial
					   trsm subproblem that solves the kc x kc lower triangular
					   block of A against the current row-panel of B. */ \
					thread_pa = bli_thrinfo_sub_prenode( thread_ic ); \
\
					/* There is no parallelism in IC loop when computing with the
					   lower triangular block of A. Thus, each thread will start at
					   ii = 0. The the IC loop bound (m_end) will change, though, if
					   and when ii reaches kc_cur (ie: moves into a rank-kc gemm
					   subproblem). */ \
				} \
				if ( ii == kc_cur ) \
				{ \
					/* Make sure all threads from the trsm subproblem have finished
					   updating the row-panel of B, since that row-panel is now an
					   input to the rank-kc gemm that follows. */ \
/*
printf( "tid %d: barrier\n", (int)bli_thread_ocomm_id( thread_ic ) ); \
*/ \
					bli_thread_barrier( thread_ic ); \
\
					/* "Dereference" the thrinfo_t pointer to the previous loop's
					   sub-node, which contains communicators and threading info for
					   rank-kc gemm that follows the initial trsm subproblem. */ \
					thread_pa = bli_thrinfo_sub_node( thread_ic ); \
\
/*
if ( bli_thread_am_ochief( thread_pa ) ) \
printf( "chief pa: ii == %d\n", (int)kc_cur ); \
*/ \
					/* Compute how much is left of the m dimension after the
					   triangular block.*/ \
					const dim_t m_ic21 = m_ic - kc_cur; \
\
					/* Compute the IC loop thread range for the current thread. */ \
					dim_t ic_start0, ic_end0; \
					bli_thread_range_sub( thread_ic, m_ic21, MR, FALSE, &ic_start0, &ic_end0 ); \
/*
printf( "tid %d: ic_start0 = %d ic_end0 = %d\n", (int)bli_thread_ocomm_id( thread_ic ), (int)ic_start0, (int)ic_end0 ); \
*/ \
\
					/* If the range is empty, then it means there isn't enough work
					   to give to all thread groups in the IC loop. In that case,
					   those threads with empty ranges sit out the rest of the
					   computation. */ \
					if ( ic_start0 == ic_end0 ) break; \
\
					/* Update the loop counter and loop bound to reflect the m dim
					   range partitioning performed above. */ \
					ii    = kc_cur + ic_start0; \
					m_end = kc_cur + ic_end0; \
				} \
/*
				else \
				{ \
					thread_pa = bli_thrinfo_sub_node( thread_ic ); \
				} \
*/ \
\
\
\
				const doff_t diagoffa_ic = diagoffa_pc + ( doff_t )ii; \
\
				/* Calculate the current IC block dimension. */ \
				if ( quadratic_mc ) \
					mc_cur = ( ii < kc_cur \
							   ? bli_trsmsup_ll_diag_mc( ii, kc_cur, diagoffa_ic, \
														 MC, MR, KC ) \
							   : bli_min( MC, m_end - ii ) \
							 ); \
				else \
					mc_cur = bli_min( MC, m_end - ii ); \
/*
				if ( ii < kc_cur ) \
				{ \
					mc_cur = bli_trsmsup_ll_diag_mc( ii, kc_cur, diagoffa_ic, MC, MR, KC ); \
				} \
				else \
				{ \
					mc_cur = bli_min( MC, m_end - ii ); \
				} \
*/ \
/*printf( "tid %d: trim mc case\n", (int)bli_thread_ocomm_id( thread_ic ) );*/ \
/*printf( "tid %d: gemm mc case\n", (int)bli_thread_ocomm_id( thread_ic ) );*/ \
				/*
				mc_cur = bli_min( MC, m_end - ii ); \
				mc_cur = ( MC <= m_end - ii ? MC : m_end - ii ); \
				*/ \
/*
printf( "tid %d: ii = %3d m_end = %d mc_cur = %d\n", (int)bli_thread_ocomm_id( thread_ic ), (int)ii, (int)m_end, (int)mc_cur ); \
*/ \
/*
*/ \
\
\
\
				ctype* restrict a_ic = a_pc0 + ii * icstep_a; \
				ctype* restrict c_ic = c_jc0 + ii * icstep_c; \
				ctype* restrict d_ic = d_use + ii * icstep_d; \
\
				ctype* a_use; \
				inc_t  rs_a_use, cs_a_use , ps_a_use; \
\
				/* "Dereference" the bszid_t array and thrinfo_t pointers to the
				   previous loop's children. Note that the thrinfo_t node will have
				   already been created by a previous call to bli_thrinfo_grow(),
				   since bszid values of BLIS_NO_PART cause the tree to grow by two
				   (e.g. to the next bszid that is a normal bszid_t value). */ \
				/*
				bszid_t* restrict bszids_pa = &bszids_ic[1]; \
				thread_pa = bli_thrinfo_sub_node( thread_ic ); \
				*/ \
\
				/* Determine the packing buffer and related parameters for matrix
				   A. (If A will not be packed, then a_use will be set to point to
				   a and the _a_use strides will be set accordingly.) Then call
				   the packm sup variant chooser. */ \
				PASTEMAC(ch,packtrim_sup_a) \
				( \
				  packtria, \
				  BLIS_BUFFER_FOR_A_BLOCK, \
				  diagoffa_ic, \
				  BLIS_LOWER, \
				  BLIS_NO_TRANSPOSE, \
				  MC,     KC, \
				  mc_cur, kc_cur, MR, \
				  &one_local, \
				  a_ic,   rs_a,      cs_a, \
				  &a_use, &rs_a_use, &cs_a_use, \
				                     &ps_a_use, \
				  cntx, \
				  rntm, \
				  &mem_a, \
				  thread_pa  \
				); \
				/*else { a_use = a_ic; rs_a_use = rs_a; \
				                     cs_a_use = cs_a; ps_a_use = irstep_a; } \
*/ \
				if ( 0 ) \
				     { a_use = a_ic; rs_a_use = rs_a; \
				                     cs_a_use = cs_a; ps_a_use = irstep_a; } \
\
				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */ \
				ctype* restrict a_ic_use = a_use; \
\
				/* Embed the panel stride of A within the auxinfo_t object. */ \
				bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
				/* Grow the thrinfo_t tree. */ \
				bszid_t* restrict bszids_jr = &bszids_pa[1]; \
				                  thread_jr = bli_thrinfo_sub_node( thread_pa ); \
/*
if ( bli_thread_am_ochief( thread_pa ) ) \
printf( "chief pa: growing jr\n" ); \
*/ \
				bli_thrinfo_sup_grow( BLIS_TRSM, rntm, bszids_jr, thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
				dim_t jr_left =   nc_cur % NR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
/*
printf( "nr_cur = %d  nc_cur = %d\n", (int)nr_cur, (int)nc_cur ); \
*/ \
\
					ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
\
					/* If fusing the packing of B, pack the current micropanel of B
					   to the appropriate spot in the packing buffer for B. */ \
					if ( ii == 0 && fuse_packb ) \
					{ \
/*
printf( "calling packb kernel, j = %d\n", (int)j ); \
*/ \
						ctype* restrict b_src_jr = b_pc + j * jrstep_b; \
\
/*PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: packed b_src_jr", kc_cur, nr_cur, b_src_jr, rs_b, cs_b, "%5.2f", "" );*/ \
						packb_ker \
						( \
						  BLIS_NO_CONJUGATE, \
						  schema_b, \
						  nr_cur, \
						  kc_cur, \
						  kc_cur, \
						  &one_local, \
						  b_src_jr, cs_b, rs_b, \
						  b_jr,           NR, /* Change this to PACKNR? */ \
						  cntx  \
						); \
/*PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: packed b_jr", kc_cur, nr_cur, b_jr, NR, 1, "%5.2f", "" );*/ \
					} \
\
\
\
					/* Compute number of primary and leftover components of the IR loop. */ \
					const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR; \
					const dim_t ir_left =   mc_cur % MR; \
\
					ctype* restrict a_ir = a_ic_use; \
					ctype* restrict c_ir = c_jr; \
					ctype* restrict d_ir = d_ic; \
\
					/* Loop over the m dimension (MR rows at a time). */ \
					for ( dim_t i = 0; i < ir_iter; i += 1 ) \
					{ \
						const dim_t mr_cur = ( bli_is_not_edge_f( i, ir_iter, ir_left ) ? MR : ir_left ); \
\
						/*ctype* restrict a_ir = a_ic_use + i * ps_a_use;*/ \
						/*ctype* restrict c_ir = c_jr     + i * irstep_c;*/ \
						/*ctype* restrict d_ir = d_ic     + i * irstep_d;*/ \
\
						const doff_t diagoffa_ir = diagoffa_ic + ( doff_t )i*MR; \
						const dim_t  k10         = bli_min( diagoffa_ir, kc_cur ); \
						const dim_t  k1011       = bli_min( diagoffa_ir + mr_cur, kc_cur ); \
\
						/* Compute the pointers to the various gemmtrsm subpartitions. */ \
						ctype* restrict a10 = a_ir; \
						ctype* restrict a11 = a_ir + k10 * cs_a_use; \
						ctype* restrict b01 = b_jr; \
						ctype* restrict b11 = b_jr + k10 * rs_b_use; \
						ctype* restrict c11 = c_ir; \
						ctype* restrict d11 = d_ir; \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: a10", mr_cur, k10, a10, rs_a,     cs_a,     "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: b01", k10, nr_cur, b01, rs_b_use, cs_b_use, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: a11", mr_cur, mr_cur, a11, rs_a,     cs_a,     "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmtrsmsup_ll_var2: b11", mr_cur, nr_cur, b11, rs_b_use, cs_b_use, "%5.2f", "" ); \
*/ \
/*
printf( "k kc_cur k10 = %d %d %d\n", (int)k, (int)kc_cur, (int)k10 ); \
*/ \
						/* Invoke the unified gemmtrsmsup/gemmsup microkernel. */ \
						gemmtrsmsup_ker \
						( \
						  mr_cur, \
						  nr_cur, \
						  k10, \
						  kc_cur, \
						  alpha_use, \
						  a10, \
						  a11, rs_a_use, cs_a_use, \
						  d11, \
						  b01, \
						  b11, rs_b_use, cs_b_use, \
						  c11, rs_b,     cs_b, \
						  &aux, \
						  cntx  \
						); \
\
						if ( 0 ) a_ir += ps_a_use; \
						else     a_ir += MR * k1011; \
						c_ir += irstep_c; \
						d_ir += irstep_d; \
					} \
				} \
\
				/* If we are fusing the packing of B, we need to use a barrier
				   after the first iteration of the IC loop to make sure that
				   all threads are finished packing before potentially allowing
				   other threads to compute with the (not-yet-completely packed)
				   packing buffer of B. */ \
				if ( ii == 0 && fuse_packb ) \
				{ \
					bli_thread_barrier( thread_ic ); \
				} \
			} \
\
			/* NOTE: This barrier is needed since we perform packing on B
			   unconditionally. (Without this barrier, a thread could race
			   ahead and begin overwriting the packing buffer with data from
			   the next rank-kc update before other threads are finished
			   with the current rank-kc update.) */ \
			if ( packb && !fuse_packb ) bli_thread_barrier( thread_pb ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTEMAC(ch,packtrim_sup_finalize_mem_a) \
	( \
	  packtria, \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTEMAC(ch,packm_sup_finalize_mem_b) \
	( \
	  packb, \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
	PASTEMAC(ch,packm_sup_finalize_mem_d) \
	( \
	  rntm, \
	  &mem_d, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( trsmsup_ll_ref_var2 )

