## tensorrt model input 
```
 input_data = {
                'image': pp_image,
                'scale_factor': pipeline_info.get('scale_factors'),
                'im_shape': pipeline_info.get('im_shape')
            }
```

