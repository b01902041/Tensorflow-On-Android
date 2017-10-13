package com.example.claire.tensorflowvdsr;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class MainActivity extends Activity {

    // For uglyman
    private static final String MODEL_FILE = "file:///android_asset/frozen_graph_float.pb";
    private static final String INPUT_NODE = "Placeholder_2";
    private static final String OUTPUT_NODE = "VDSR/VDSR_var_scope/VDSR_1/clip_by_value";

    // For VDSR
    /*
    private static final String MODEL_FILE = "file:///android_asset/optimized_tfdroid.pb";
    private static final String INPUT_NODE = "t_image_input_to_SRGAN_generator";
    private static final String OUTPUT_NODE = "SRGAN_vdsr_acid/out";
    */

    private int inputSize_y = 512;
    private int inputSize_x = 768;
    private int[] intValues = new int[inputSize_x * inputSize_y];
    private float[] floatValues = new float[inputSize_x * inputSize_y * 3];

    private float[] output = new float[inputSize_x * inputSize_y * 3];
    private int[] output_intValues = new int[inputSize_x * inputSize_y];

    private ScaleImage imageView;
    private ImageView imageView2;

    public TensorFlowInferenceInterface inferenceInterface;

    public Bitmap bitmap;
    public Uri uri;

    static {
        System.loadLibrary("tensorflow_inference");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);

        // Choose image from cell phone--------------------------------------
        //找尋Button按鈕
        Button button1 = (Button)findViewById(R.id.button1);
        //設定按鈕監聽式
        button1.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent();
                //開啟Pictures畫面Type設定為image
                intent.setType("image/*");
                //使用Intent.ACTION_GET_CONTENT這個Action//會開啟選取圖檔視窗讓您選取手機內圖檔
                intent.setAction(Intent.ACTION_GET_CONTENT);
                //取得相片後返回本畫面
                startActivityForResult(intent, 1);
            }
        });

        Button button2 = (Button)findViewById(R.id.button2);
        //設定按鈕監聽式
        button2.setOnClickListener(new Button.OnClickListener() {
            public void onClick(View v) {
                System.out.println("Start run");
                Run();
            }
        });

        //----------------------------------------------------------------------------
    }

    //取得相片後返回的監聽式
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //當使用者按下確定後
        if (resultCode == RESULT_OK) {
            //取得圖檔的路徑位置
            uri = data.getData();
            //寫log
            Log.e("uri", uri.toString());
            //抽象資料的接口
            ContentResolver cr = this.getContentResolver();
            try {
                //由抽象資料接口轉換圖檔路徑為Bitmap
                bitmap = BitmapFactory.decodeStream(cr.openInputStream(uri));
                //取得圖片控制項ImageView
                imageView = (ScaleImage) findViewById(R.id.imageView);
                //scaleImage = (ScaleImage) findViewById(R.id.scale_imageView);
                // 將Bitmap設定到ImageView
                imageView.setImageBitmap(bitmap);
            } catch (FileNotFoundException e) {
                Log.e("Exception", e.getMessage(),e);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    public void Run(){
        long startTime = System.nanoTime();

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        // For uglyman

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3] = ((intValues[i] >> 16) & 0xFF) / 1.0f;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 1.0f;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 1.0f;
            //System.out.println("input is");
            //System.out.println(floatValues[i * 3 + 0]);
        }

        // For VDSR
        /*
        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3] = ((intValues[i] >> 16) & 0xFF) / 127.5f - 1;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 127.5f - 1;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 127.5f - 1;
            //System.out.println("input is");
            //System.out.println(floatValues[i * 3 + 0]);
        }
        */

        inferenceInterface.feed(INPUT_NODE, floatValues, 1, inputSize_y, inputSize_x, 3);
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE, output);

        // For uglyman

        for (int i = 0; i < output_intValues.length; ++i) {
            //System.out.println("origin output is");
            //System.out.println(output[i * 3 + 0]);
            output[i * 3] = output[i * 3] * 255.0f;
            output[i * 3 + 1] = output[i * 3 + 1] * 255.0f;
            output[i * 3 + 2] = output[i * 3 + 2] * 255.0f;
            //System.out.println("now output is");
            //System.out.println(output[i * 3 + 0]);
        }


        // For VDSR
        /*
        for (int i = 0; i < output_intValues.length; ++i) {
            //System.out.println("origin output is");
            //System.out.println(output[i * 3 + 0]);
            output[i * 3] = (output[i * 3] + 1 ) * 127.5f;
            output[i * 3 + 1] = (output[i * 3 + 1] + 1) * 127.5f;
            output[i * 3 + 2] = (output[i * 3 + 2] + 1) * 127.5f;
            //System.out.println("now output is");
            //System.out.println(output[i * 3 + 0]);
        }
        */

        //System.out.println("last is");
        //System.out.println(output[3]);


        for (int i = 0; i < output_intValues.length; ++i) {
            output_intValues[i] = 0xff000000 | ((int)(output[i * 3])<< 16) | ((int)(output[i * 3 + 1])<< 8) | (int)(output[i * 3 + 2]);
        }
        Bitmap bitmap2 = Bitmap.createBitmap(output_intValues, inputSize_x, inputSize_y, Bitmap.Config.ARGB_8888);
        // show gray image
        imageView2 = (ImageView)findViewById(R.id.imageView2);
        imageView2.setImageBitmap(bitmap2);

        try {
            // 取得外部儲存裝置路徑
            String path = Environment.getExternalStorageDirectory().toString ();
            // 開啟檔案
            File file = new File( path, Double.toString(startTime).concat(".png"));
            // 開啟檔案串流
            FileOutputStream out = new FileOutputStream(file);
            // 將 Bitmap壓縮成指定格式的圖片並寫入檔案串流
            bitmap2.compress ( Bitmap. CompressFormat.PNG , 100 , out);
            // 刷新並關閉檔案串流
            out.flush ();
            out.close ();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace ();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace ();
        }
        long stopTime = System.nanoTime();
        double seconds = (double)(stopTime - startTime) / 1000000000.0;
        System.out.println(seconds);
        String time = Double.toString(seconds);
        TextView text = (TextView)findViewById(R.id.text);
        text.setText(time);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}