/* Squared distance to an interpolation triangle and coordinate descent on
   four-atom supports. Coefficient space has the Frobenius inner product. */
#include <math.h>
#include <string.h>
#define N 7
typedef struct {
  double v[4];
} Vec;
static Vec add(Vec a, Vec b, double k) {
  for (int j = 0; j < 4; j++)
    a.v[j] += k * b.v[j];
  return a;
}
static double dot(Vec a, Vec b) {
  double s = 0;
  for (int j = 0; j < 4; j++)
    s += a.v[j] * b.v[j];
  return s;
}
static Vec scale(Vec a, double c) {
  for (int j = 0; j < 4; j++)
    a.v[j] *= c;
  return a;
}
static double clip(double x) { return fmax(0., fmin(1., x)); }
static double segment(Vec a, Vec b) {
  Vec d = add(b, a, -1.);
  Vec v = add(a, d, clip(-dot(a, d) / fmax(1e-100, dot(d, d))));
  return dot(v, v);
}
static double triangle(Vec a, Vec b, Vec c) {
  Vec u = add(b, a, -1), v = add(c, a, -1);
  double A = dot(u, u), B = dot(v, v), D = dot(u, v), e = -dot(u, a),
         f = -dot(v, a);
  double det = A * B - D * D, x = (B * e - D * f) / det,
         y = (A * f - D * e) / det;
  if (det > 0 && x >= 0 && y >= 0 && x + y <= 1) {
    Vec r = add(add(a, u, x), v, y);
    return dot(r, r);
  }
  return fmin(segment(a, b), fmin(segment(a, c), segment(b, c)));
}
double cost(int n, int k, const double *X, const double *Y, int passes) {
  Vec edges[N][N] = {0};
  double ds[N][N] = {0};
  double best = 1e100;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++) {
      double h = X[2 * i] * X[2 * j + 1] - X[2 * i + 1] * X[2 * j];
      if (fabs(h) < 1e-12)
        continue;
      ds[i][j] = ds[j][i] = h * h;
      Vec b;
      for (int m = 0; m < 2; m++) {
        b.v[2 * m] =
            (Y[2 * i + m] * X[2 * j + 1] - Y[2 * j + m] * X[2 * i + 1]) / h;
        b.v[2 * m + 1] =
            (X[2 * i] * Y[2 * j + m] - X[2 * j] * Y[2 * i + m]) / h;
      }
      edges[i][j] = edges[j][i] = b;
      best = fmin(best, dot(b, b));
    }
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      for (int l = j + 1; l < n; l++) {
        Vec a[3];
        int m = 0;
        if (ds[i][j])
          a[m++] = edges[i][j];
        if (ds[i][l])
          a[m++] = edges[i][l];
        if (ds[j][l])
          a[m++] = edges[j][l];
        if (m == 2)
          best = fmin(best, segment(a[0], a[1]));
        if (m == 3)
          best = fmin(best, triangle(a[0], a[1], a[2]));
      }
  if (k == 3)
    return best;
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      for (int l = j + 1; l < n; l++)
        for (int m = l + 1; m < n; m++) {
          int ids[4] = {i, j, l, m};
          double w[4] = {1, 1, 1, 1};
          for (int it = 0; it < 1 + 4 * passes; it++) {
            int h = it % 4;
            Vec b = {0}, v = {0};
            double s = 0, d = 0;
            for (int a = 0; a < 4; a++)
              for (int c = a + 1; c < 4; c++) {
                double e = ds[ids[a]][ids[c]];
                if (a == h || c == h) {
                  e *= w[a == h ? c : a];
                  d += e;
                  v = add(v, edges[ids[a]][ids[c]], e);
                } else {
                  e *= w[a] * w[c];
                  s += e;
                  b = add(b, edges[ids[a]][ids[c]], e);
                }
              }
            if (s == 0 || d == 0)
              continue;
            b = scale(b, 1 / s);
            v = scale(v, 1 / d);
            Vec diff = add(v, b, -1);
            double t = clip(-dot(b, diff) / fmax(1e-100, dot(diff, diff)));
            Vec opt = add(b, diff, t);
            best = fmin(best, dot(opt, opt));
            t = fmax(1e-7, fmin(1 - 1e-7, t));
            w[h] = fmax(1e-10, fmin(1e10, t / (1 - t) * s / d));
          }
        }
  return best;
}
/* Per-support witnesses for the outer epigraph optimizer. */
void faces(int n, int k, const double *x, const double *y, int passes,
           double *out) {
  for (int mask = 0; mask < (1 << n); mask++) {
    double a[8], b[8];
    int count = 0;
    if (__builtin_popcount((unsigned)mask) != k)
      continue;
    for (int i = 0; i < n; i++)
      if (mask & (1 << i)) {
        memcpy(a + 2 * count, x + 2 * i, 2 * sizeof(double));
        memcpy(b + 2 * count, y + 2 * i, 2 * sizeof(double));
        count++;
      }
    *out++ = cost(k, k, a, b, passes);
  }
}

/* Test the restriction to the original row multipliers. */
double unmodified(int n, int k, const double *x, const double *y, double *out) {
  double best = 1e100;
  for (int mask = 0; mask < (1 << n); ++mask) {
    int size = __builtin_popcount((unsigned)mask);
    if (size < 2 || size > k)
      continue;
    Vec numerator = {0};
    double denominator = 0;
    for (int i = 0; i < n; i++)
      if (mask & (1 << i)) {
        for (int j = i + 1; j < n; j++)
          if (mask & (1 << j)) {
            double det = x[2 * i] * x[2 * j + 1] - x[2 * i + 1] * x[2 * j];
            if (fabs(det) < 1e-12)
              continue;
            denominator += det * det;
            for (int m = 0; m < 2; m++) {
              numerator.v[2 * m] += det * (y[2 * i + m] * x[2 * j + 1] -
                                           y[2 * j + m] * x[2 * i + 1]);
              numerator.v[2 * m + 1] +=
                  det * (x[2 * i] * y[2 * j + m] - x[2 * j] * y[2 * i + m]);
            }
          }
      }
    double c = 1e100;
    if (denominator > 0) {
      Vec b = scale(numerator, 1 / denominator);
      c = dot(b, b);
    }
    *out++ = c;
    best = fmin(best, c);
  }
  return best;
}
