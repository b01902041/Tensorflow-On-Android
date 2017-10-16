package com.example.claire.tensorflowvdsr;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.drawable.Drawable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.widget.ImageView;


/**
 * Created by Claire on 2017/10/11.
 */
public class ScaleImage extends ImageView {
    private Matrix matrix = new Matrix();

    // mode can be in one of these 3 states
    private static final int NONE = 0;
    private static final int DRAG = 1;
    private static final int ZOOM = 2;
    private int mode = NONE;

    private static final int CLICK = 3;

    // Remember some things
    private PointF last = new PointF();
    private PointF start = new PointF();
    private float minScale = 1f;// default
    private float maxScale = 3f;// default
    private float minScaleTemp;// measure
    private float maxScaleTemp;// measure
    private float[] m;
    private float redundantXSpace, redundantYSpace;
    private float width, height;
    private float nowScale = 1f;
    private float origWidth, origHeight, imageWidth, imageHeight,
            redundantWidth, redundantHeight;
    private boolean fit = false;

    private long lastClickTime = 0;
    private long down_time = 0;

    private ScaleGestureDetector mScaleDetector;

    public ScaleImage(Context context) {
        super(context);
        initStanScalableImageView(context);
    }

    public ScaleImage(Context context, AttributeSet attrs) {
        super(context, attrs);
        initStanScalableImageView(context);
    }

    public ScaleImage(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
        initStanScalableImageView(context);
    }

    private void initStanScalableImageView(Context context) {
        super.setScaleType(ScaleType.CENTER);
        mScaleDetector = new ScaleGestureDetector(context, new ScaleListener());
        matrix.setTranslate(1f, 1f);
        m = new float[9];
        setImageMatrix(matrix);
        setScaleType(ScaleType.MATRIX);
        setOnTouchListener(new DragListener());
    }

    @Override
    public void setImageBitmap(Bitmap bitmap) {
        super.setImageBitmap(bitmap);
        imageWidth = bitmap.getWidth();
        imageHeight = bitmap.getHeight();
    }

    @Override
    public void setImageDrawable(Drawable drawable) {
        super.setImageDrawable(drawable);
        imageWidth = drawable.getIntrinsicWidth();
        imageHeight = drawable.getIntrinsicHeight();
    }

    @Override
    public void setImageResource(int resourceID) {
        super.setImageResource(resourceID);
        imageWidth = getResources().getDrawable(resourceID).getIntrinsicWidth();
        imageHeight = getResources().getDrawable(resourceID).getIntrinsicHeight();
    }

    public void setMaxZoom(float x) {
        this.maxScale = x;
    }

    public void setMinZoom(float x) {
        this.minScale = x;
    }

    public void setFit(boolean fit) {
        this.fit = fit;
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec);
        width = MeasureSpec.getSize(widthMeasureSpec);
        height = MeasureSpec.getSize(heightMeasureSpec);
        float scale;
        if (fit) {
            float scaleX = (float) width / (float) imageWidth;
            float scaleY = (float) height / (float) imageHeight;
            scale = Math.min(scaleX, scaleY);
            matrix.setScale(scale, scale);
        } else {
            scale = 1;
        }
        minScaleTemp = minScale * scale;
        maxScaleTemp = maxScale * scale;

        setImageMatrix(matrix);
        nowScale = scale;

        // Center the image

        redundantHeight = (scale * (float) imageHeight);
        redundantWidth = (scale * (float) imageWidth);
        redundantYSpace = (float) height - (scale * (float) imageHeight);
        redundantXSpace = (float) width - redundantWidth;
        redundantYSpace /= (float) 2;
        redundantXSpace /= (float) 2;

        matrix.getValues(m);
        float x = m[Matrix.MTRANS_X];
        float y = m[Matrix.MTRANS_Y];
        matrix.postTranslate(redundantXSpace - x, redundantYSpace - y);

        origWidth = width - 2 * redundantXSpace;
        origHeight = height - 2 * redundantYSpace;
        setImageMatrix(matrix);

    }

    private class DragListener implements OnTouchListener {

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            mScaleDetector.onTouchEvent(event);
            matrix.getValues(m);
            float x = m[Matrix.MTRANS_X];
            float y = m[Matrix.MTRANS_Y];
            PointF curr = new PointF(event.getX(), event.getY());

            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    last.set(event.getX(), event.getY());
                    start.set(last);
                    if (nowScale * imageWidth > width * 1.05
                            || nowScale * imageHeight > height * 1.05) {
                        mode = DRAG;
                    }
                    break;
                case MotionEvent.ACTION_MOVE:
                    float deltaX = curr.x - last.x;
                    float deltaY = curr.y - last.y;
                    if (mode == DRAG) {
                        float scaleWidth = Math.round(imageWidth * nowScale);
                        float scaleHeight = Math.round(imageHeight * nowScale);
                        // x
                        if (scaleWidth < width) {// 不超過邊框時置中
                            deltaX = redundantXSpace - x
                                    - (imageWidth * nowScale - redundantWidth) / 2;
                        } else {
                            if (deltaX > 0 && x + deltaX > 0) {// 往右拉&&左邊拉過頭
                                deltaX = -x;
                            } else if (deltaX < 0
                                    && x + deltaX + scaleWidth < width) {// 往左拉&&右邊拉過頭
                                deltaX = width - x - scaleWidth;
                            }
                        }
                        // y
                        if (scaleHeight < height) {// 不超過邊框時置中
                            deltaY = redundantYSpace - y
                                    - (imageHeight * nowScale - redundantHeight)
                                    / 2;
                        } else {
                            if (deltaY > 0 && y + deltaY > 0) {// 往下拉&&上邊拉過頭
                                deltaY = -y;
                            } else if (deltaY < 0
                                    && y + deltaY + scaleHeight < height) {// 往上拉&&下邊拉過頭
                                deltaY = height - y - scaleHeight;
                            }
                        }
                        matrix.postTranslate(deltaX, deltaY);
                        last.set(curr.x, curr.y);
                    }
                    break;

                case MotionEvent.ACTION_UP:
                    mode = NONE;
                    int xDiff = (int) Math.abs(curr.x - start.x);
                    int yDiff = (int) Math.abs(curr.y - start.y);
                    if (xDiff < CLICK && yDiff < CLICK)
                        performClick();
                    break;

                case MotionEvent.ACTION_POINTER_UP:
                    Log.d("Scale", "up");
                    // Double click , add by claire----------------------------------------------
                    /*
                    if ( mode == DRAG) {

                        Log.d("Scale", "double");
                        long now = System.currentTimeMillis();

                        if ((now - lastClickTime) < 300) {
                            Log.d("Scale", "double < 300");
                            matrix.setScale(minScale, minScale);
                            now = 0;
                        }

                        lastClickTime = now;

                    }
                    */

                    //---------------------------------------------------------------------------
                    mode = NONE;
                    break;
            }
            setImageMatrix(matrix);
            invalidate();
            return true;
        }
    }

    private class ScaleListener extends
            ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScaleBegin(ScaleGestureDetector detector) {
            mode = ZOOM;
            last.set(detector.getFocusX(), detector.getFocusY());
            start.set(last);
            return true;
        }

        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            float mScaleFactor = detector.getScaleFactor();
            float origScale = nowScale;
            nowScale *= mScaleFactor;
            if (nowScale > maxScaleTemp) {
                nowScale = maxScaleTemp;
                mScaleFactor = maxScaleTemp / origScale;
            } else if (nowScale < minScaleTemp) {
                nowScale = minScaleTemp;
                mScaleFactor = minScaleTemp / origScale;
            }
            if (origWidth * nowScale <= width
                    || origHeight * nowScale <= height) {
                matrix.postScale(mScaleFactor, mScaleFactor, width / 2,
                        height / 2);
            } else {
                matrix.postScale(mScaleFactor, mScaleFactor,
                        detector.getFocusX(), detector.getFocusY());
            }
            matrix.getValues(m);
            float x = m[Matrix.MTRANS_X];
            float y = m[Matrix.MTRANS_Y];
            float scaleWidth = Math.round(imageWidth * nowScale);
            float scaleHeight = Math.round(imageHeight * nowScale);
            if (nowScale < origScale) {// 縮小時置中
                if (x > 0 && x + scaleWidth > width) {// 只有左邊有空隙
                    matrix.postTranslate(-x, 0);
                } else if (x < 0 && x + scaleWidth < width) {// 只有右邊有空隙
                    matrix.postTranslate(width - x - scaleWidth, 0);
                } else if (x > 0 && x + scaleWidth < width) {// 兩邊都有空隙
                    matrix.postTranslate((width - scaleWidth) / 2 - x, 0);
                }
                if (y > 0 && y + scaleHeight > height) {// 只有上邊有空隙
                    matrix.postTranslate(0, -y);
                } else if (y < 0 && y + scaleHeight < height) {// 只有下邊有空隙
                    matrix.postTranslate(0, height - y - scaleHeight);
                } else if (y > 0 && y + scaleHeight < height) {// 兩邊都有空隙
                    matrix.postTranslate(0, (height - scaleHeight) / 2 - y);
                }
            }

            PointF curr = new PointF(detector.getFocusX(), detector.getFocusY());
            float deltaX = curr.x - last.x;
            float deltaY = curr.y - last.y;
            last.set(curr.x, curr.y);
            // x
            if (scaleWidth < width) {// 不超過邊框時置中
                deltaX = redundantXSpace - x
                        - (imageWidth * nowScale - redundantWidth) / 2;
                return true;// 縮放這邊已經置中過了，跳調
            } else {
                if (deltaX > 0 && x + deltaX > 0) {// 往右拉&&左邊拉過頭
                    deltaX = -x;
                } else if (deltaX < 0 && x + deltaX + scaleWidth < width) {// 往左拉&&右邊拉過頭
                    deltaX = width - x - scaleWidth;
                }
            }
            // y
            if (scaleHeight < height) {// 不超過邊框時置中
                deltaY = redundantYSpace - y
                        - (imageHeight * nowScale - redundantHeight) / 2;
                return true;// 縮放這邊已經置中過了，跳調
            } else {
                if (deltaY > 0 && y + deltaY > 0) {// 往下拉&&上邊拉過頭
                    deltaY = -y;
                } else if (deltaY < 0 && y + deltaY + scaleHeight < height) {// 往上拉&&下邊拉過頭
                    deltaY = height - y - scaleHeight;
                }
            }
            matrix.postTranslate(deltaX, deltaY);
            return true;
        }
    }
}
