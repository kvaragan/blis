/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2020, Advanced Micro Devices, Inc.

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
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t            n_max, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	/* Check the mem_t entry provided by the caller. If it is unallocated,
	   then we need to acquire a block from the memory broker. */ \
	if ( bli_mem_is_unalloc( mem ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			/* Compute the size of the memory block eneded. */ \
			const siz_t size_needed = sizeof( ctype ) * n_max; \
\
			/* Since there is not yet a dedicated pool of buffers for
			   diagonals, allocate from the heap. */ \
			const packbuf_t pack_buf_type = BLIS_BUFFER_FOR_GEN_USE; \
\
			/* Acquire directly to the chief thread's mem_t that was
			   passed in. It needs to be that mem_t struct, and not a
			   local (temporary) mem_t, since there is no barrier until
			   after packing is finished, which could allow a race
			   condition whereby the chief thread exits the current
			   function before the other threads have a chance to copy
			   from it. (A barrier would fix that race condition, but
			   then again, I prefer to keep barriers to a minimum.) */ \
			bli_membrk_acquire_m \
			( \
			  rntm, \
			  size_needed, \
			  pack_buf_type, \
			  mem  \
			); \
		} \
\
		/* Broadcast the address of the chief thread's passed-in mem_t
		   to all threads. */ \
		mem_t* mem_p = bli_thread_broadcast( thread, mem ); \
\
		/* Non-chief threads: Copy the contents of the chief thread's
		   passed-in mem_t to the passed-in mem_t for this thread. (The
		   chief thread already has the mem_t, so it does not need to
		   perform any copy.) */ \
		if ( !bli_thread_am_ochief( thread ) ) \
		{ \
			*mem = *mem_p; \
		} \
	} \
	else /* if ( bli_mem_is_alloc( mem ) ) */ \
	{ \
		/* If the mem_t entry provided by the caller does NOT contain a NULL
		   buffer, then a block has already been acquired from the memory
		   broker and cached by the caller. */ \
\
		/* Compute the size of the memory block eneded. */ \
		const siz_t size_needed = sizeof( ctype ) * n_max; \
\
		/* As a sanity check, we should make sure that the mem_t object isn't
		   associated with a block that is too small compared to the size of
		   the packed matrix buffer that is needed, according to the value
		   computed above. */ \
		const siz_t mem_size = bli_mem_size( mem ); \
\
		if ( mem_size < size_needed ) \
		{ \
			printf( "bli_packm_sup_init_mem_d: mem_t size is too small!\n" ); \
			bli_abort(); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_init_mem_d )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	if ( thread != NULL ) \
	if ( bli_thread_am_ochief( thread ) ) \
	{ \
		/* Check the mem_t entry provided by the caller. Only proceed if it
		   is allocated, which it should be. */ \
		if ( bli_mem_is_alloc( mem ) ) \
		{ \
			bli_membrk_release \
			( \
			  rntm, \
			  mem \
			); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( packm_sup_finalize_mem_d )


//
// Define BLAS-like interfaces to the front-facing packing function.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       diag_t           diaga, \
       dim_t            k_alloc, \
       dim_t            k, \
       ctype*  restrict a, inc_t rs_a, inc_t cs_a, \
       ctype** restrict p, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Barrier to make sure all threads are caught up and ready to begin
	   the packm stage. */ \
	bli_thread_barrier( thread ); \
\
	/* Check the mem_t entry provided by the caller. If it is unallocated,
	   then we need to acquire a block from the memory broker. */ \
	if ( bli_mem_is_unalloc( mem ) ) \
	{ \
		/* Acquire a mem_t entry for the packing destination buffer and
		   broadcast it to all threads. */ \
		PASTEMAC(ch,packm_sup_init_mem_d) \
		( \
		  k_alloc, \
		  rntm, \
		  mem, \
		  thread  \
		); \
	} \
\
	/* Query the packing buffer from the mem_t entry acquired above
	   and save it to the pointer variable p that was passed in. */ \
	*p = bli_mem_buffer( mem ); \
\
	/* Since it's such a small amount of data we're packing, let the
	   chief thread handle it all. */ \
	if ( bli_thread_am_ochief( thread ) ) \
	{ \
		/* The increment between diagonal elements of A is equal to the sum of
		   the row and column strides. The increment to use in the packed
		   diagonal vector is always unit. */ \
		const inc_t inca = rs_a + cs_a; \
		const inc_t incd = 1; \
\
		/* Query the copyv kernel function pointer from the context. */ \
/*
		PASTECH2(ch,copyv,_ker_ft) \
		f = bli_cntx_get_l1v_ker_dt( dt, BLIS_COPYV_KER, cntx ); \
\
		f \
		( \
		  BLIS_NO_CONJUGATE, \
		  k, \
		  a,  inca, \
		  *p, incp, \
		  cntx \
		); \
*/ \
/*
		ctype* restrict a_i = a; \
		ctype* restrict d_i = *p; \
\
		for ( dim_t i = 0; i < k; ++i ) \
		{ \
			PASTEMAC(ch,invert2s)( *a_i, *d_i ); \
\
			PASTEMAC(ch,copys)( *a_i, *d_i ); \
			PASTEMAC(ch,inverts)( *d_i ); \
\
			a_i += inca; \
			d_i += incd; \
		} \
*/ \
/*
		ctype* restrict d = *p; \
\
		for ( dim_t i = 0; i < k; ++i ) \
		{ \
			ctype* restrict a_i = a + i*inca; \
			ctype* restrict d_i = d + i*incd; \
\
			PASTEMAC(ch,invert2s)( *a_i, *d_i ); \
		} \
*/ \
		if ( bli_is_nonunit_diag( diaga ) ) \
		{ \
			ctype* restrict a_i = a; \
			ctype* restrict d_i = *p; \
\
			for ( dim_t i = 0; i < k; ++i ) \
			{ \
				PASTEMAC(ch,copys)( *a_i, *d_i ); \
				PASTEMAC(ch,inverts)( *d_i ); \
\
				a_i += inca; \
				d_i += incd; \
			} \
		} \
		else /* if ( bli_is_unit_diag( diaga ) ) */ \
		{ \
			PASTECH2(ch,setv,_ker_ft) \
			f = bli_cntx_get_l1v_ker_dt( dt, BLIS_SETV_KER, cntx ); \
\
			ctype* restrict d     = *p; \
			ctype* restrict one_p = PASTEMAC(ch,1); \
\
			f \
			( \
			  BLIS_NO_CONJUGATE, \
			  k, \
			  one_p, \
			  d, incd, \
			  cntx \
			); \
		} \
	} \
\
	/* Barrier so that packing is done before computation. */ \
	bli_thread_barrier( thread ); \
}

INSERT_GENTFUNC_BASIC0( packm_sup_d )

