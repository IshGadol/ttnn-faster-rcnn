# faster r-cnn → tt-nn layer mapping (draft)

- backbone (resnet50): conv/bn/relu → ttnn conv2d + elemwise
- fpn: lateral + top-down merges → conv + add + relu
- rpn: conv + objectness/reg heads
- roi align/pool: (ttnn op or custom kernel TBD)
- box head: fc/relu layers
- cls/reg heads: fc layers

## open items
- roi align availability / fallback
- nms op location (host vs device)
