/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"

#ifdef BLIS_ENABLE_BLAS2BLIS

#include "bl2_f2c.h"

/* ctbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(c,tbmv)(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, complex *a, integer *lda, complex *x, integer *incx) 
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    complex q__1, q__2, q__3;

    /* Builtin functions */
    void r_cnjg(complex *, complex *);

    /* Local variables */
    integer info;
    complex temp;
    integer i__, j, l;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kplus1, ix, jx, kx = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  CTBMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular band matrix, with ( k + 1 ) diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := conjg( A' )*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - COMPLEX          array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX          array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! lsame_(diag, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("CTBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = lsame_(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = lsame_(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX   too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*         Form  x := A*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].r != 0.f || x[i__2].i != 0.f) {
			i__2 = j;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			l = kplus1 - j;
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__5 = l + i__ + j * a_dim1;
			    q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    q__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    q__1.r = x[i__3].r + q__2.r, q__1.i = x[i__3].i + 
				    q__2.i;
			    x[i__2].r = q__1.r, x[i__2].i = q__1.i;
/* L10: */
			}
			if (nounit) {
			    i__4 = j;
			    i__2 = j;
			    i__3 = kplus1 + j * a_dim1;
			    q__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
				    i__3].i, q__1.i = x[i__2].r * a[i__3].i + 
				    x[i__2].i * a[i__3].r;
			    x[i__4].r = q__1.r, x[i__4].i = q__1.i;
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__4 = jx;
		    if (x[i__4].r != 0.f || x[i__4].i != 0.f) {
			i__4 = jx;
			temp.r = x[i__4].r, temp.i = x[i__4].i;
			ix = kx;
			l = kplus1 - j;
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
			    i__4 = ix;
			    i__2 = ix;
			    i__5 = l + i__ + j * a_dim1;
			    q__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    q__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    q__1.r = x[i__2].r + q__2.r, q__1.i = x[i__2].i + 
				    q__2.i;
			    x[i__4].r = q__1.r, x[i__4].i = q__1.i;
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__3 = jx;
			    i__4 = jx;
			    i__2 = kplus1 + j * a_dim1;
			    q__1.r = x[i__4].r * a[i__2].r - x[i__4].i * a[
				    i__2].i, q__1.i = x[i__4].r * a[i__2].i + 
				    x[i__4].i * a[i__2].r;
			    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
			}
		    }
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].r != 0.f || x[i__1].i != 0.f) {
			i__1 = j;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			l = 1 - j;
/* Computing MIN */
			i__1 = *n, i__3 = j + *k;
			i__4 = j + 1;
			for (i__ = min(i__1,i__3); i__ >= i__4; --i__) {
			    i__1 = i__;
			    i__3 = i__;
			    i__2 = l + i__ + j * a_dim1;
			    q__2.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    q__2.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    q__1.r = x[i__3].r + q__2.r, q__1.i = x[i__3].i + 
				    q__2.i;
			    x[i__1].r = q__1.r, x[i__1].i = q__1.i;
/* L50: */
			}
			if (nounit) {
			    i__4 = j;
			    i__1 = j;
			    i__3 = j * a_dim1 + 1;
			    q__1.r = x[i__1].r * a[i__3].r - x[i__1].i * a[
				    i__3].i, q__1.i = x[i__1].r * a[i__3].i + 
				    x[i__1].i * a[i__3].r;
			    x[i__4].r = q__1.r, x[i__4].i = q__1.i;
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__4 = jx;
		    if (x[i__4].r != 0.f || x[i__4].i != 0.f) {
			i__4 = jx;
			temp.r = x[i__4].r, temp.i = x[i__4].i;
			ix = kx;
			l = 1 - j;
/* Computing MIN */
			i__4 = *n, i__1 = j + *k;
			i__3 = j + 1;
			for (i__ = min(i__4,i__1); i__ >= i__3; --i__) {
			    i__4 = ix;
			    i__1 = ix;
			    i__2 = l + i__ + j * a_dim1;
			    q__2.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    q__2.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    q__1.r = x[i__1].r + q__2.r, q__1.i = x[i__1].i + 
				    q__2.i;
			    x[i__4].r = q__1.r, x[i__4].i = q__1.i;
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__3 = jx;
			    i__4 = jx;
			    i__1 = j * a_dim1 + 1;
			    q__1.r = x[i__4].r * a[i__1].r - x[i__4].i * a[
				    i__1].i, q__1.i = x[i__4].r * a[i__1].i + 
				    x[i__4].i * a[i__1].r;
			    x[i__3].r = q__1.r, x[i__3].i = q__1.i;
			}
		    }
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__3 = j;
		    temp.r = x[i__3].r, temp.i = x[i__3].i;
		    l = kplus1 - j;
		    if (noconj) {
			if (nounit) {
			    i__3 = kplus1 + j * a_dim1;
			    q__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, 
				    q__1.i = temp.r * a[i__3].i + temp.i * a[
				    i__3].r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__1 = i__;
			    q__2.r = a[i__4].r * x[i__1].r - a[i__4].i * x[
				    i__1].i, q__2.i = a[i__4].r * x[i__1].i + 
				    a[i__4].i * x[i__1].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
/* L90: */
			}
		    } else {
			if (nounit) {
			    r_cnjg(&q__2, &a[kplus1 + j * a_dim1]);
			    q__1.r = temp.r * q__2.r - temp.i * q__2.i, 
				    q__1.i = temp.r * q__2.i + temp.i * 
				    q__2.r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__4 = i__;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i, 
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
/* L100: */
			}
		    }
		    i__3 = j;
		    x[i__3].r = temp.r, x[i__3].i = temp.i;
/* L110: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__3 = jx;
		    temp.r = x[i__3].r, temp.i = x[i__3].i;
		    kx -= *incx;
		    ix = kx;
		    l = kplus1 - j;
		    if (noconj) {
			if (nounit) {
			    i__3 = kplus1 + j * a_dim1;
			    q__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, 
				    q__1.i = temp.r * a[i__3].i + temp.i * a[
				    i__3].r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__1 = ix;
			    q__2.r = a[i__4].r * x[i__1].r - a[i__4].i * x[
				    i__1].i, q__2.i = a[i__4].r * x[i__1].i + 
				    a[i__4].i * x[i__1].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
			    ix -= *incx;
/* L120: */
			}
		    } else {
			if (nounit) {
			    r_cnjg(&q__2, &a[kplus1 + j * a_dim1]);
			    q__1.r = temp.r * q__2.r - temp.i * q__2.i, 
				    q__1.i = temp.r * q__2.i + temp.i * 
				    q__2.r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__4 = ix;
			    q__2.r = q__3.r * x[i__4].r - q__3.i * x[i__4].i, 
				    q__2.i = q__3.r * x[i__4].i + q__3.i * x[
				    i__4].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
			    ix -= *incx;
/* L130: */
			}
		    }
		    i__3 = jx;
		    x[i__3].r = temp.r, x[i__3].i = temp.i;
		    jx -= *incx;
/* L140: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    i__4 = j;
		    temp.r = x[i__4].r, temp.i = x[i__4].i;
		    l = 1 - j;
		    if (noconj) {
			if (nounit) {
			    i__4 = j * a_dim1 + 1;
			    q__1.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    q__1.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__2 = i__;
			    q__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, q__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
/* L150: */
			}
		    } else {
			if (nounit) {
			    r_cnjg(&q__2, &a[j * a_dim1 + 1]);
			    q__1.r = temp.r * q__2.r - temp.i * q__2.i, 
				    q__1.i = temp.r * q__2.i + temp.i * 
				    q__2.r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__1 = i__;
			    q__2.r = q__3.r * x[i__1].r - q__3.i * x[i__1].i, 
				    q__2.i = q__3.r * x[i__1].i + q__3.i * x[
				    i__1].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
/* L160: */
			}
		    }
		    i__4 = j;
		    x[i__4].r = temp.r, x[i__4].i = temp.i;
/* L170: */
		}
	    } else {
		jx = kx;
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    i__4 = jx;
		    temp.r = x[i__4].r, temp.i = x[i__4].i;
		    kx += *incx;
		    ix = kx;
		    l = 1 - j;
		    if (noconj) {
			if (nounit) {
			    i__4 = j * a_dim1 + 1;
			    q__1.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    q__1.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__2 = ix;
			    q__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, q__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
			    ix += *incx;
/* L180: */
			}
		    } else {
			if (nounit) {
			    r_cnjg(&q__2, &a[j * a_dim1 + 1]);
			    q__1.r = temp.r * q__2.r - temp.i * q__2.i, 
				    q__1.i = temp.r * q__2.i + temp.i * 
				    q__2.r;
			    temp.r = q__1.r, temp.i = q__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    r_cnjg(&q__3, &a[l + i__ + j * a_dim1]);
			    i__1 = ix;
			    q__2.r = q__3.r * x[i__1].r - q__3.i * x[i__1].i, 
				    q__2.i = q__3.r * x[i__1].i + q__3.i * x[
				    i__1].r;
			    q__1.r = temp.r + q__2.r, q__1.i = temp.i + 
				    q__2.i;
			    temp.r = q__1.r, temp.i = q__1.i;
			    ix += *incx;
/* L190: */
			}
		    }
		    i__4 = jx;
		    x[i__4].r = temp.r, x[i__4].i = temp.i;
		    jx += *incx;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of CTBMV . */

} /* ctbmv_ */

/* dtbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(d,tbmv)(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer info;
    doublereal temp;
    integer i__, j, l;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kplus1, ix, jx, kx = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DTBMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular band matrix, with ( k + 1 ) diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := A'*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - DOUBLE PRECISION array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! lsame_(diag, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DTBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = lsame_(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX   too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*         Form  x := A*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.) {
			temp = x[j];
			l = kplus1 - j;
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
			    x[i__] += temp * a[l + i__ + j * a_dim1];
/* L10: */
			}
			if (nounit) {
			    x[j] *= a[kplus1 + j * a_dim1];
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			l = kplus1 - j;
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
			    x[ix] += temp * a[l + i__ + j * a_dim1];
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    x[jx] *= a[kplus1 + j * a_dim1];
			}
		    }
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.) {
			temp = x[j];
			l = 1 - j;
/* Computing MIN */
			i__1 = *n, i__3 = j + *k;
			i__4 = j + 1;
			for (i__ = min(i__1,i__3); i__ >= i__4; --i__) {
			    x[i__] += temp * a[l + i__ + j * a_dim1];
/* L50: */
			}
			if (nounit) {
			    x[j] *= a[j * a_dim1 + 1];
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.) {
			temp = x[jx];
			ix = kx;
			l = 1 - j;
/* Computing MIN */
			i__4 = *n, i__1 = j + *k;
			i__3 = j + 1;
			for (i__ = min(i__4,i__1); i__ >= i__3; --i__) {
			    x[ix] += temp * a[l + i__ + j * a_dim1];
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    x[jx] *= a[j * a_dim1 + 1];
			}
		    }
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    l = kplus1 - j;
		    if (nounit) {
			temp *= a[kplus1 + j * a_dim1];
		    }
/* Computing MAX */
		    i__4 = 1, i__1 = j - *k;
		    i__3 = max(i__4,i__1);
		    for (i__ = j - 1; i__ >= i__3; --i__) {
			temp += a[l + i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    kx -= *incx;
		    ix = kx;
		    l = kplus1 - j;
		    if (nounit) {
			temp *= a[kplus1 + j * a_dim1];
		    }
/* Computing MAX */
		    i__4 = 1, i__1 = j - *k;
		    i__3 = max(i__4,i__1);
		    for (i__ = j - 1; i__ >= i__3; --i__) {
			temp += a[l + i__ + j * a_dim1] * x[ix];
			ix -= *incx;
/* L110: */
		    }
		    x[jx] = temp;
		    jx -= *incx;
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    temp = x[j];
		    l = 1 - j;
		    if (nounit) {
			temp *= a[j * a_dim1 + 1];
		    }
/* Computing MIN */
		    i__1 = *n, i__2 = j + *k;
		    i__4 = min(i__1,i__2);
		    for (i__ = j + 1; i__ <= i__4; ++i__) {
			temp += a[l + i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		jx = kx;
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    temp = x[jx];
		    kx += *incx;
		    ix = kx;
		    l = 1 - j;
		    if (nounit) {
			temp *= a[j * a_dim1 + 1];
		    }
/* Computing MIN */
		    i__1 = *n, i__2 = j + *k;
		    i__4 = min(i__1,i__2);
		    for (i__ = j + 1; i__ <= i__4; ++i__) {
			temp += a[l + i__ + j * a_dim1] * x[ix];
			ix += *incx;
/* L150: */
		    }
		    x[jx] = temp;
		    jx += *incx;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of DTBMV . */

} /* dtbmv_ */

/* stbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(s,tbmv)(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, real *a, integer *lda, real *x, integer *incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer info;
    real temp;
    integer i__, j, l;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kplus1, ix, jx, kx = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  STBMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular band matrix, with ( k + 1 ) diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := A'*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - REAL             array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! lsame_(diag, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("STBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    nounit = lsame_(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX   too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*         Form  x := A*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			l = kplus1 - j;
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
			    x[i__] += temp * a[l + i__ + j * a_dim1];
/* L10: */
			}
			if (nounit) {
			    x[j] *= a[kplus1 + j * a_dim1];
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			l = kplus1 - j;
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
			    x[ix] += temp * a[l + i__ + j * a_dim1];
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    x[jx] *= a[kplus1 + j * a_dim1];
			}
		    }
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			l = 1 - j;
/* Computing MIN */
			i__1 = *n, i__3 = j + *k;
			i__4 = j + 1;
			for (i__ = min(i__1,i__3); i__ >= i__4; --i__) {
			    x[i__] += temp * a[l + i__ + j * a_dim1];
/* L50: */
			}
			if (nounit) {
			    x[j] *= a[j * a_dim1 + 1];
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			l = 1 - j;
/* Computing MIN */
			i__4 = *n, i__1 = j + *k;
			i__3 = j + 1;
			for (i__ = min(i__4,i__1); i__ >= i__3; --i__) {
			    x[ix] += temp * a[l + i__ + j * a_dim1];
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    x[jx] *= a[j * a_dim1 + 1];
			}
		    }
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    l = kplus1 - j;
		    if (nounit) {
			temp *= a[kplus1 + j * a_dim1];
		    }
/* Computing MAX */
		    i__4 = 1, i__1 = j - *k;
		    i__3 = max(i__4,i__1);
		    for (i__ = j - 1; i__ >= i__3; --i__) {
			temp += a[l + i__ + j * a_dim1] * x[i__];
/* L90: */
		    }
		    x[j] = temp;
/* L100: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    kx -= *incx;
		    ix = kx;
		    l = kplus1 - j;
		    if (nounit) {
			temp *= a[kplus1 + j * a_dim1];
		    }
/* Computing MAX */
		    i__4 = 1, i__1 = j - *k;
		    i__3 = max(i__4,i__1);
		    for (i__ = j - 1; i__ >= i__3; --i__) {
			temp += a[l + i__ + j * a_dim1] * x[ix];
			ix -= *incx;
/* L110: */
		    }
		    x[jx] = temp;
		    jx -= *incx;
/* L120: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    temp = x[j];
		    l = 1 - j;
		    if (nounit) {
			temp *= a[j * a_dim1 + 1];
		    }
/* Computing MIN */
		    i__1 = *n, i__2 = j + *k;
		    i__4 = min(i__1,i__2);
		    for (i__ = j + 1; i__ <= i__4; ++i__) {
			temp += a[l + i__ + j * a_dim1] * x[i__];
/* L130: */
		    }
		    x[j] = temp;
/* L140: */
		}
	    } else {
		jx = kx;
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    temp = x[jx];
		    kx += *incx;
		    ix = kx;
		    l = 1 - j;
		    if (nounit) {
			temp *= a[j * a_dim1 + 1];
		    }
/* Computing MIN */
		    i__1 = *n, i__2 = j + *k;
		    i__4 = min(i__1,i__2);
		    for (i__ = j + 1; i__ <= i__4; ++i__) {
			temp += a[l + i__ + j * a_dim1] * x[ix];
			ix += *incx;
/* L150: */
		    }
		    x[jx] = temp;
		    jx += *incx;
/* L160: */
		}
	    }
	}
    }

    return 0;

/*     End of STBMV . */

} /* stbmv_ */

/* ztbmv.f -- translated by f2c (version 19991025).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

/* Subroutine */ int PASTEF77(z,tbmv)(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublecomplex *a, integer *lda, doublecomplex *x, integer 
	*incx)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;
    doublecomplex z__1, z__2, z__3;

    /* Builtin functions */
    void d_cnjg(doublecomplex *, doublecomplex *);

    /* Local variables */
    integer info;
    doublecomplex temp;
    integer i__, j, l;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    integer kplus1, ix, jx, kx = 0;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    logical noconj, nounit;

/*     .. Scalar Arguments .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  ZTBMV  performs one of the matrix-vector operations */

/*     x := A*x,   or   x := A'*x,   or   x := conjg( A' )*x, */

/*  where x is an n element vector and  A is an n by n unit, or non-unit, */
/*  upper or lower triangular band matrix, with ( k + 1 ) diagonals. */

/*  Parameters */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On entry, UPLO specifies whether the matrix is an upper or */
/*           lower triangular matrix as follows: */

/*              UPLO = 'U' or 'u'   A is an upper triangular matrix. */

/*              UPLO = 'L' or 'l'   A is a lower triangular matrix. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry, TRANS specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   x := A*x. */

/*              TRANS = 'T' or 't'   x := A'*x. */

/*              TRANS = 'C' or 'c'   x := conjg( A' )*x. */

/*           Unchanged on exit. */

/*  DIAG   - CHARACTER*1. */
/*           On entry, DIAG specifies whether or not A is unit */
/*           triangular as follows: */

/*              DIAG = 'U' or 'u'   A is assumed to be unit triangular. */

/*              DIAG = 'N' or 'n'   A is not assumed to be unit */
/*                                  triangular. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry, N specifies the order of the matrix A. */
/*           N must be at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with UPLO = 'U' or 'u', K specifies the number of */
/*           super-diagonals of the matrix A. */
/*           On entry with UPLO = 'L' or 'l', K specifies the number of */
/*           sub-diagonals of the matrix A. */
/*           K must satisfy  0 .le. K. */
/*           Unchanged on exit. */

/*  A      - COMPLEX*16       array of DIMENSION ( LDA, n ). */
/*           Before entry with UPLO = 'U' or 'u', the leading ( k + 1 ) */
/*           by n part of the array A must contain the upper triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row */
/*           ( k + 1 ) of the array, the first super-diagonal starting at */
/*           position 2 in row k, and so on. The top left k by k triangle */
/*           of the array A is not referenced. */
/*           The following program segment will transfer an upper */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = K + 1 - J */
/*                    DO 10, I = MAX( 1, J - K ), J */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Before entry with UPLO = 'L' or 'l', the leading ( k + 1 ) */
/*           by n part of the array A must contain the lower triangular */
/*           band part of the matrix of coefficients, supplied column by */
/*           column, with the leading diagonal of the matrix in row 1 of */
/*           the array, the first sub-diagonal starting at position 1 in */
/*           row 2, and so on. The bottom right k by k triangle of the */
/*           array A is not referenced. */
/*           The following program segment will transfer a lower */
/*           triangular band matrix from conventional full matrix storage */
/*           to band storage: */

/*                 DO 20, J = 1, N */
/*                    M = 1 - J */
/*                    DO 10, I = J, MIN( N, J + K ) */
/*                       A( M + I, J ) = matrix( I, J ) */
/*              10    CONTINUE */
/*              20 CONTINUE */

/*           Note that when DIAG = 'U' or 'u' the elements of the array A */
/*           corresponding to the diagonal elements of the matrix are not */
/*           referenced, but are assumed to be unity. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in the calling (sub) program. LDA must be at least */
/*           ( k + 1 ). */
/*           Unchanged on exit. */

/*  X      - COMPLEX*16       array of dimension at least */
/*           ( 1 + ( n - 1 )*abs( INCX ) ). */
/*           Before entry, the incremented array X must contain the n */
/*           element vector x. On exit, X is overwritten with the */
/*           tranformed vector x. */

/*  INCX   - INTEGER. */
/*           On entry, INCX specifies the increment for the elements of */
/*           X. INCX must not be zero. */
/*           Unchanged on exit. */


/*  Level 2 Blas routine. */

/*  -- Written on 22-October-1986. */
/*     Jack Dongarra, Argonne National Lab. */
/*     Jeremy Du Croz, Nag Central Office. */
/*     Sven Hammarling, Nag Central Office. */
/*     Richard Hanson, Sandia National Labs. */


/*     .. Parameters .. */
/*     .. Local Scalars .. */
/*     .. External Functions .. */
/*     .. External Subroutines .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --x;

    /* Function Body */
    info = 0;
    if (! lsame_(uplo, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(uplo, "L", (
	    ftnlen)1, (ftnlen)1)) {
	info = 1;
    } else if (! lsame_(trans, "N", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, 
	    "T", (ftnlen)1, (ftnlen)1) && ! lsame_(trans, "C", (ftnlen)1, (
	    ftnlen)1)) {
	info = 2;
    } else if (! lsame_(diag, "U", (ftnlen)1, (ftnlen)1) && ! lsame_(diag, 
	    "N", (ftnlen)1, (ftnlen)1)) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < *k + 1) {
	info = 7;
    } else if (*incx == 0) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("ZTBMV ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    }

    noconj = lsame_(trans, "T", (ftnlen)1, (ftnlen)1);
    nounit = lsame_(diag, "N", (ftnlen)1, (ftnlen)1);

/*     Set up the start point in X if the increment is not unity. This */
/*     will be  ( N - 1 )*INCX   too small for descending loops. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are */
/*     accessed sequentially with one pass through A. */

    if (lsame_(trans, "N", (ftnlen)1, (ftnlen)1)) {

/*         Form  x := A*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    if (x[i__2].r != 0. || x[i__2].i != 0.) {
			i__2 = j;
			temp.r = x[i__2].r, temp.i = x[i__2].i;
			l = kplus1 - j;
/* Computing MAX */
			i__2 = 1, i__3 = j - *k;
			i__4 = j - 1;
			for (i__ = max(i__2,i__3); i__ <= i__4; ++i__) {
			    i__2 = i__;
			    i__3 = i__;
			    i__5 = l + i__ + j * a_dim1;
			    z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    z__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__2].r = z__1.r, x[i__2].i = z__1.i;
/* L10: */
			}
			if (nounit) {
			    i__4 = j;
			    i__2 = j;
			    i__3 = kplus1 + j * a_dim1;
			    z__1.r = x[i__2].r * a[i__3].r - x[i__2].i * a[
				    i__3].i, z__1.i = x[i__2].r * a[i__3].i + 
				    x[i__2].i * a[i__3].r;
			    x[i__4].r = z__1.r, x[i__4].i = z__1.i;
			}
		    }
/* L20: */
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__4 = jx;
		    if (x[i__4].r != 0. || x[i__4].i != 0.) {
			i__4 = jx;
			temp.r = x[i__4].r, temp.i = x[i__4].i;
			ix = kx;
			l = kplus1 - j;
/* Computing MAX */
			i__4 = 1, i__2 = j - *k;
			i__3 = j - 1;
			for (i__ = max(i__4,i__2); i__ <= i__3; ++i__) {
			    i__4 = ix;
			    i__2 = ix;
			    i__5 = l + i__ + j * a_dim1;
			    z__2.r = temp.r * a[i__5].r - temp.i * a[i__5].i, 
				    z__2.i = temp.r * a[i__5].i + temp.i * a[
				    i__5].r;
			    z__1.r = x[i__2].r + z__2.r, z__1.i = x[i__2].i + 
				    z__2.i;
			    x[i__4].r = z__1.r, x[i__4].i = z__1.i;
			    ix += *incx;
/* L30: */
			}
			if (nounit) {
			    i__3 = jx;
			    i__4 = jx;
			    i__2 = kplus1 + j * a_dim1;
			    z__1.r = x[i__4].r * a[i__2].r - x[i__4].i * a[
				    i__2].i, z__1.i = x[i__4].r * a[i__2].i + 
				    x[i__4].i * a[i__2].r;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
			}
		    }
		    jx += *incx;
		    if (j > *k) {
			kx += *incx;
		    }
/* L40: */
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__1 = j;
		    if (x[i__1].r != 0. || x[i__1].i != 0.) {
			i__1 = j;
			temp.r = x[i__1].r, temp.i = x[i__1].i;
			l = 1 - j;
/* Computing MIN */
			i__1 = *n, i__3 = j + *k;
			i__4 = j + 1;
			for (i__ = min(i__1,i__3); i__ >= i__4; --i__) {
			    i__1 = i__;
			    i__3 = i__;
			    i__2 = l + i__ + j * a_dim1;
			    z__2.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    z__2.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    z__1.r = x[i__3].r + z__2.r, z__1.i = x[i__3].i + 
				    z__2.i;
			    x[i__1].r = z__1.r, x[i__1].i = z__1.i;
/* L50: */
			}
			if (nounit) {
			    i__4 = j;
			    i__1 = j;
			    i__3 = j * a_dim1 + 1;
			    z__1.r = x[i__1].r * a[i__3].r - x[i__1].i * a[
				    i__3].i, z__1.i = x[i__1].r * a[i__3].i + 
				    x[i__1].i * a[i__3].r;
			    x[i__4].r = z__1.r, x[i__4].i = z__1.i;
			}
		    }
/* L60: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__4 = jx;
		    if (x[i__4].r != 0. || x[i__4].i != 0.) {
			i__4 = jx;
			temp.r = x[i__4].r, temp.i = x[i__4].i;
			ix = kx;
			l = 1 - j;
/* Computing MIN */
			i__4 = *n, i__1 = j + *k;
			i__3 = j + 1;
			for (i__ = min(i__4,i__1); i__ >= i__3; --i__) {
			    i__4 = ix;
			    i__1 = ix;
			    i__2 = l + i__ + j * a_dim1;
			    z__2.r = temp.r * a[i__2].r - temp.i * a[i__2].i, 
				    z__2.i = temp.r * a[i__2].i + temp.i * a[
				    i__2].r;
			    z__1.r = x[i__1].r + z__2.r, z__1.i = x[i__1].i + 
				    z__2.i;
			    x[i__4].r = z__1.r, x[i__4].i = z__1.i;
			    ix -= *incx;
/* L70: */
			}
			if (nounit) {
			    i__3 = jx;
			    i__4 = jx;
			    i__1 = j * a_dim1 + 1;
			    z__1.r = x[i__4].r * a[i__1].r - x[i__4].i * a[
				    i__1].i, z__1.i = x[i__4].r * a[i__1].i + 
				    x[i__4].i * a[i__1].r;
			    x[i__3].r = z__1.r, x[i__3].i = z__1.i;
			}
		    }
		    jx -= *incx;
		    if (*n - j >= *k) {
			kx -= *incx;
		    }
/* L80: */
		}
	    }
	}
    } else {

/*        Form  x := A'*x  or  x := conjg( A' )*x. */

	if (lsame_(uplo, "U", (ftnlen)1, (ftnlen)1)) {
	    kplus1 = *k + 1;
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    i__3 = j;
		    temp.r = x[i__3].r, temp.i = x[i__3].i;
		    l = kplus1 - j;
		    if (noconj) {
			if (nounit) {
			    i__3 = kplus1 + j * a_dim1;
			    z__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, 
				    z__1.i = temp.r * a[i__3].i + temp.i * a[
				    i__3].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__1 = i__;
			    z__2.r = a[i__4].r * x[i__1].r - a[i__4].i * x[
				    i__1].i, z__2.i = a[i__4].r * x[i__1].i + 
				    a[i__4].i * x[i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L90: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a[kplus1 + j * a_dim1]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__4 = i__;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, 
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L100: */
			}
		    }
		    i__3 = j;
		    x[i__3].r = temp.r, x[i__3].i = temp.i;
/* L110: */
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    i__3 = jx;
		    temp.r = x[i__3].r, temp.i = x[i__3].i;
		    kx -= *incx;
		    ix = kx;
		    l = kplus1 - j;
		    if (noconj) {
			if (nounit) {
			    i__3 = kplus1 + j * a_dim1;
			    z__1.r = temp.r * a[i__3].r - temp.i * a[i__3].i, 
				    z__1.i = temp.r * a[i__3].i + temp.i * a[
				    i__3].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    i__4 = l + i__ + j * a_dim1;
			    i__1 = ix;
			    z__2.r = a[i__4].r * x[i__1].r - a[i__4].i * x[
				    i__1].i, z__2.i = a[i__4].r * x[i__1].i + 
				    a[i__4].i * x[i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ix -= *incx;
/* L120: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a[kplus1 + j * a_dim1]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MAX */
			i__4 = 1, i__1 = j - *k;
			i__3 = max(i__4,i__1);
			for (i__ = j - 1; i__ >= i__3; --i__) {
			    d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__4 = ix;
			    z__2.r = z__3.r * x[i__4].r - z__3.i * x[i__4].i, 
				    z__2.i = z__3.r * x[i__4].i + z__3.i * x[
				    i__4].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ix -= *incx;
/* L130: */
			}
		    }
		    i__3 = jx;
		    x[i__3].r = temp.r, x[i__3].i = temp.i;
		    jx -= *incx;
/* L140: */
		}
	    }
	} else {
	    if (*incx == 1) {
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    i__4 = j;
		    temp.r = x[i__4].r, temp.i = x[i__4].i;
		    l = 1 - j;
		    if (noconj) {
			if (nounit) {
			    i__4 = j * a_dim1 + 1;
			    z__1.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    z__1.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__2 = i__;
			    z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, z__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L150: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a[j * a_dim1 + 1]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__1 = i__;
			    z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i, 
				    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
				    i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
/* L160: */
			}
		    }
		    i__4 = j;
		    x[i__4].r = temp.r, x[i__4].i = temp.i;
/* L170: */
		}
	    } else {
		jx = kx;
		i__3 = *n;
		for (j = 1; j <= i__3; ++j) {
		    i__4 = jx;
		    temp.r = x[i__4].r, temp.i = x[i__4].i;
		    kx += *incx;
		    ix = kx;
		    l = 1 - j;
		    if (noconj) {
			if (nounit) {
			    i__4 = j * a_dim1 + 1;
			    z__1.r = temp.r * a[i__4].r - temp.i * a[i__4].i, 
				    z__1.i = temp.r * a[i__4].i + temp.i * a[
				    i__4].r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    i__1 = l + i__ + j * a_dim1;
			    i__2 = ix;
			    z__2.r = a[i__1].r * x[i__2].r - a[i__1].i * x[
				    i__2].i, z__2.i = a[i__1].r * x[i__2].i + 
				    a[i__1].i * x[i__2].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ix += *incx;
/* L180: */
			}
		    } else {
			if (nounit) {
			    d_cnjg(&z__2, &a[j * a_dim1 + 1]);
			    z__1.r = temp.r * z__2.r - temp.i * z__2.i, 
				    z__1.i = temp.r * z__2.i + temp.i * 
				    z__2.r;
			    temp.r = z__1.r, temp.i = z__1.i;
			}
/* Computing MIN */
			i__1 = *n, i__2 = j + *k;
			i__4 = min(i__1,i__2);
			for (i__ = j + 1; i__ <= i__4; ++i__) {
			    d_cnjg(&z__3, &a[l + i__ + j * a_dim1]);
			    i__1 = ix;
			    z__2.r = z__3.r * x[i__1].r - z__3.i * x[i__1].i, 
				    z__2.i = z__3.r * x[i__1].i + z__3.i * x[
				    i__1].r;
			    z__1.r = temp.r + z__2.r, z__1.i = temp.i + 
				    z__2.i;
			    temp.r = z__1.r, temp.i = z__1.i;
			    ix += *incx;
/* L190: */
			}
		    }
		    i__4 = jx;
		    x[i__4].r = temp.r, x[i__4].i = temp.i;
		    jx += *incx;
/* L200: */
		}
	    }
	}
    }

    return 0;

/*     End of ZTBMV . */

} /* ztbmv_ */

#endif
