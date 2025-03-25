#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#define X_DIRECTION 0
#define Y_DIRECTION 1

/* Image structure: width, height, maximum gray value, and data as float */
typedef struct {
    int width;
    int height;
    int maxval;
    float *data;
} Image;

/* Create an image and allocate memory */
Image* createImage(int width, int height) {
    Image* img = (Image*) malloc(sizeof(Image));
    if (!img) {
        fprintf(stderr, "Error allocating image structure.\n");
        exit(1);
    }
    img->width = width;
    img->height = height;
    img->maxval = 255;
    img->data = (float*) calloc(width * height, sizeof(float));
    if (!img->data) {
        fprintf(stderr, "Error allocating image data.\n");
        exit(1);
    }
    return img;
}

/* Free an image */
void freeImage(Image* img) {
    if (img) {
        if (img->data)
            free(img->data);
        free(img);
    }
}

/* Read a PGM image (supports binary P5 and ASCII P2) */
Image* readPGM(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Cannot open file");
        exit(1);
    }

    char format[3];
    if (fscanf(fp, "%2s", format) != 1) {
        fprintf(stderr, "Error reading file format.\n");
        exit(1);
    }
    int isBinary = (format[1] == '5');

    // Skip whitespace and comments
    int c = fgetc(fp);
    while (isspace(c))
        c = fgetc(fp);
    if (c == '#') {
        while (c != '\n' && c != EOF)
            c = fgetc(fp);
    }
    while (isspace(c))
        c = fgetc(fp);
    ungetc(c, fp);

    int width, height, maxval;
    if (fscanf(fp, "%d %d %d", &width, &height, &maxval) != 3) {
        fprintf(stderr, "Error reading image size or max value.\n");
        exit(1);
    }
    fgetc(fp); // Consume newline after header

    Image* img = createImage(width, height);
    img->maxval = maxval;
    int size = width * height;

    if (isBinary) {
        unsigned char *temp = (unsigned char*) malloc(size);
        if (!temp) {
            fprintf(stderr, "Memory allocation error.\n");
            exit(1);
        }
        if (fread(temp, 1, size, fp) != size) {
            fprintf(stderr, "Error reading image data.\n");
            exit(1);
        }
        for (int i = 0; i < size; i++) {
            img->data[i] = (float) temp[i];
        }
        free(temp);
    } else {
        for (int i = 0; i < size; i++) {
            int pixel;
            fscanf(fp, "%d", &pixel);
            img->data[i] = (float) pixel;
        }
    }
    fclose(fp);
    return img;
}

/* Write a PGM image (binary P5); scales data to 0–255 */
void writePGM(const char* filename, Image* img) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Cannot open output file");
        exit(1);
    }
    fprintf(fp, "P5\n%d %d\n255\n", img->width, img->height);
    int size = img->width * img->height;
    float minVal = img->data[0], maxVal = img->data[0];
    for (int i = 0; i < size; i++) {
        if (img->data[i] < minVal) minVal = img->data[i];
        if (img->data[i] > maxVal) maxVal = img->data[i];
    }
    float range = maxVal - minVal;
    if (range == 0) range = 1;
    unsigned char* out = (unsigned char*) malloc(size);
    if (!out) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        float val = (img->data[i] - minVal) / range * 255.0f;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        out[i] = (unsigned char) val;
    }
    fwrite(out, 1, size, fp);
    free(out);
    fclose(fp);
}

/* Part One: Gaussian smoothing using a separable kernel */
Image* gaussianSmooth(Image *in, double sigma) {
    int radius = (int) ceil(3 * sigma);
    int kernelSize = 2 * radius + 1;
    double *kernel = (double*) malloc(kernelSize * sizeof(double));
    if (!kernel) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    double sum = 0.0;
    for (int i = 0; i < kernelSize; i++) {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }

    int width = in->width, height = in->height;
    Image* temp = createImage(width, height);
    // Horizontal convolution
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; k++) {
                int ix = x + k;
                if (ix < 0) ix = 0;
                if (ix >= width) ix = width - 1;
                acc += in->data[y * width + ix] * kernel[k + radius];
            }
            temp->data[y * width + x] = acc;
        }
    }
    // Vertical convolution
    Image* out = createImage(width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double acc = 0.0;
            for (int k = -radius; k <= radius; k++) {
                int iy = y + k;
                if (iy < 0) iy = 0;
                if (iy >= height) iy = height - 1;
                acc += temp->data[iy * width + x] * kernel[k + radius];
            }
            out->data[y * width + x] = acc;
        }
    }
    free(kernel);
    freeImage(temp);
    return out;
}

/* Part One: Sobel filtering to compute gradients.
   direction: X_DIRECTION or Y_DIRECTION */
Image* applySobel(Image *img, int direction) {
    int width = img->width, height = img->height;
    Image* out = createImage(width, height);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum = 0.0f;
            if (direction == X_DIRECTION) {
                // Sobel X kernel:
                //  -1  0  1
                //  -2  0  2
                //  -1  0  1
                sum += -1 * img->data[(y - 1) * width + (x - 1)];
                sum +=  0 * img->data[(y - 1) * width + x];
                sum +=  1 * img->data[(y - 1) * width + (x + 1)];
                sum += -2 * img->data[y * width + (x - 1)];
                sum +=  0 * img->data[y * width + x];
                sum +=  2 * img->data[y * width + (x + 1)];
                sum += -1 * img->data[(y + 1) * width + (x - 1)];
                sum +=  0 * img->data[(y + 1) * width + x];
                sum +=  1 * img->data[(y + 1) * width + (x + 1)];
            } else {
                // Sobel Y kernel:
                //  -1 -2 -1
                //   0  0  0
                //   1  2  1
                sum += -1 * img->data[(y - 1) * width + (x - 1)];
                sum += -2 * img->data[(y - 1) * width + x];
                sum += -1 * img->data[(y - 1) * width + (x + 1)];
                sum +=  0 * img->data[y * width + (x - 1)];
                sum +=  0 * img->data[y * width + x];
                sum +=  0 * img->data[y * width + (x + 1)];
                sum +=  1 * img->data[(y + 1) * width + (x - 1)];
                sum +=  2 * img->data[(y + 1) * width + x];
                sum +=  1 * img->data[(y + 1) * width + (x + 1)];
            }
            out->data[y * width + x] = sum;
        }
    }
    return out;
}

/* Compute the gradient magnitude from gx and gy */
Image* computeMagnitude(Image* gx, Image* gy) {
    int width = gx->width, height = gx->height;
    Image* mag = createImage(width, height);
    int size = width * height;
    for (int i = 0; i < size; i++) {
        mag->data[i] = sqrt(gx->data[i]*gx->data[i] + gy->data[i]*gy->data[i]);
    }
    return mag;
}

/* Non-Maximum Suppression (Quantized Version)
   This function implements the quantized peak detection as given in your snippet.
   It uses the gradient magnitude (mag) and the gradients (gx, gy).
*/
Image* nonMaximumSuppressionQuantized(Image* mag, Image* gx, Image* gy) {
    int width = mag->width, height = mag->height;
    Image* cand = createImage(width, height);
    int MR = 1; // Margin to avoid border issues
    for (int i = MR; i < height - MR; i++) {
        for (int j = MR; j < width - MR; j++) {
            int idx = i * width + j;
            float currentMag = mag->data[idx];
            float gx_val = gx->data[idx];
            if (fabs(gx_val) < 1e-6)
                gx_val = 0.00001f;
            float slope = gy->data[idx] / gx_val;
            cand->data[idx] = 0; // Default to non-edge
            // Near-horizontal edges (slope near 0)
            if ((slope <= 0.4142f) && (slope > -0.4142f)) {
                if ((currentMag > mag->data[i * width + (j - 1)]) &&
                    (currentMag > mag->data[i * width + (j + 1)]))
                    cand->data[idx] = 255;
            }
            // Positive diagonal edges (slope between 0.4142 and 2.4142)
            else if ((slope <= 2.4142f) && (slope > 0.4142f)) {
                if ((currentMag > mag->data[(i - 1) * width + (j - 1)]) &&
                    (currentMag > mag->data[(i + 1) * width + (j + 1)]))
                    cand->data[idx] = 255;
            }
            // Negative diagonal edges (slope between -2.4142 and -0.4142)
            else if ((slope <= -0.4142f) && (slope > -2.4142f)) {
                if ((currentMag > mag->data[(i + 1) * width + (j - 1)]) &&
                    (currentMag > mag->data[(i - 1) * width + (j + 1)]))
                    cand->data[idx] = 255;
            }
            // Otherwise, treat as vertical edges (steep slopes)
            else {
                if ((currentMag > mag->data[(i - 1) * width + j]) &&
                    (currentMag > mag->data[(i + 1) * width + j]))
                    cand->data[idx] = 255;
            }
        }
    }
    // Set border pixels to zero
    for (int j = 0; j < width; j++) {
        cand->data[j] = 0;
        cand->data[(height - 1) * width + j] = 0;
    }
    for (int i = 0; i < height; i++) {
        cand->data[i * width] = 0;
        cand->data[i * width + (width - 1)] = 0;
    }
    return cand;
}

/* Hysteresis Thresholding:
   Marks strong edges (where candidate value is 255 and mag >= hi),
   then propagates using 8-connected neighbors where mag >= lo.
*/
Image* hysteresisThresholding(Image* nms, Image* mag, double hi, double lo) {
    int width = nms->width, height = nms->height;
    Image* edges = createImage(width, height);
    int size = width * height;
    int *stackX = (int*) malloc(size * sizeof(int));
    int *stackY = (int*) malloc(size * sizeof(int));
    int stackSize = 0;

    // Mark strong edges and add them to the stack
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            if (nms->data[idx] > 0 && mag->data[idx] >= hi) {
                edges->data[idx] = 255;
                stackX[stackSize] = j;
                stackY[stackSize] = i;
                stackSize++;
            }
        }
    }

    // Propagate connectivity
    while (stackSize > 0) {
        stackSize--;
        int x = stackX[stackSize];
        int y = stackY[stackSize];
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int nx = x + j, ny = y + i;
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                int nidx = ny * width + nx;
                if (edges->data[nidx] == 0 &&
                    nms->data[nidx] > 0 &&
                    mag->data[nidx] >= lo) {
                    edges->data[nidx] = 255;
                    stackX[stackSize] = nx;
                    stackY[stackSize] = ny;
                    stackSize++;
                }
            }
        }
    }
    free(stackX);
    free(stackY);
    return edges;
}

/* Comparator for qsort for floats */
int compareFloats(const void* a, const void* b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}

/* Compute thresholds automatically.
   It sorts the gradient magnitudes and chooses hi so that only a given fraction (percent) of pixels are above it.
   Then, lo is set to 25% of hi.
   Adjustments: percent is increased and low threshold multiplier is reduced.
*/
void computeThresholds(Image* mag, double percent, double* hi, double* lo) {
    int size = mag->width * mag->height;
    float *array = (float*) malloc(size * sizeof(float));
    if (!array) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        array[i] = mag->data[i];
    }
    qsort(array, size, sizeof(float), compareFloats);
    int index = (int)((1 - percent) * size);
    if (index < 0) index = 0;
    if (index >= size) index = size - 1;
    *hi = array[index];
    *lo = *hi * 0.2;  // Lower multiplier: 20% of hi to capture smaller edges
    free(array);
}

/* Main function: uses hard-coded parameters so it runs without command-line arguments.
   It reads "garb34.pgm", applies Gaussian smoothing, computes gradients,
   uses quantized non-maximum suppression, and then performs hysteresis thresholding.
   Three output files are generated: "magnitude.pgm", "peaks.pgm", and "FinalEdges.pgm".
*/
int main(void) {
    const char *input_filename = "garb34.pgm";
    const double sigma = 1.0;
    // Increase the fraction so more pixels qualify as strong edges
    const double percent = 0.015;  // Changed from 0.01 to 0.02
    double hi, lo;

    Image* inputImage = readPGM(input_filename);

    /* Part One: Gaussian smoothing and gradient computation */
    Image* smoothed = gaussianSmooth(inputImage, sigma);
    Image* gx = applySobel(smoothed, X_DIRECTION);
    Image* gy = applySobel(smoothed, Y_DIRECTION);
    Image* magnitude = computeMagnitude(gx, gy);
    writePGM("magnitude.pgm", magnitude);

    /* Part Two: Non-Maximum Suppression using quantized (slope-based) method */
    Image* peaks = nonMaximumSuppressionQuantized(magnitude, gx, gy);
    writePGM("peaks.pgm", peaks);

    /* Part Three: Hysteresis Thresholding */
    computeThresholds(magnitude, percent, &hi, &lo);
    printf("Computed thresholds: hi = %f, lo = %f\n", hi, lo);
    Image* finalEdges = hysteresisThresholding(peaks, magnitude, hi, lo);
    writePGM("FinalEdges.pgm", finalEdges);

    /* Cleanup */
    freeImage(inputImage);
    freeImage(smoothed);
    freeImage(gx);
    freeImage(gy);
    freeImage(magnitude);
    freeImage(peaks);
    freeImage(finalEdges);

    return 0;
}
