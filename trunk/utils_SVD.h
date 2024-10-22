/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class utils_SVD */

#ifndef _Included_utils_SVD
#define _Included_utils_SVD
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     utils_SVD
 * Method:    powerMethod
 * Signature: ([I[I[D[D[D)D
 */
JNIEXPORT jdouble JNICALL Java_utils_SVD_powerMethod
  (JNIEnv *, jclass, jintArray, jintArray, jdoubleArray, jdoubleArray, jdoubleArray);

/*
 * Class:     utils_SVD
 * Method:    lowRankSvd
 * Signature: ([D[DIII[D[D[D)I
 */
JNIEXPORT jint JNICALL Java_utils_SVD_lowRankSvd
  (JNIEnv *, jclass, jdoubleArray, jdoubleArray, jint, jint, jint, jdoubleArray, jdoubleArray, jdoubleArray);

#ifdef __cplusplus
}
#endif
#endif
