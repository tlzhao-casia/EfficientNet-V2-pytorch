# This is the ALCS implementation of efficientnet v2

Training efficientnetv2 involves many many tricks, I will implement and learn them.

## Update at 2022.06.10
I have made a stupid bug. I forgot to set the model to eval mode during evaluation. I have fixed it. I am still training the model now.

## Update at 2022.06.09
I have implemented the progressive training techniques: Progressively increase both the image size and regularization magnitudes during training. The relative hyper-parameters are:

<style>
th {
	background-color: #D6EEEE;
    text-align: center
}
</style>
<table>
<caption> Hyper-parameters for progressive training </caption>
<tr>
<th rowspan=2> Model </th> <th colspan=2> S </th> 
</tr>
<tr>
<th> Start </th> <th> End </th>
</tr>
<tr>
<td> Image size </td> <td> 128 </td> <td> 300 </td>
</tr>
<tr>
<td> RandAugment </td> <td> 5 </td> <td> 15 </td>
</tr>
<tr>
<td> Mixup alpha </td> <td> 0 </td> <td> 0 </td>
</tr>
<tr>
<td> Dropout rate </td> <td> 0.1 </td> <td> 0.3 </td>
</tr>
</table>

## Update at 2022.06.08
Network architecture implementation of EfficientNetV2-S completed.

| Model | Params | FLOPS | Acc. |
| ----- | ------ | ----- | ---- |
| EfficientNet-V2 | 22.103832M | | |
