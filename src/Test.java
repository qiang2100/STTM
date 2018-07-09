/**
 * Created by jipengqiang on 18/7/4.
 */


import java.util.HashMap;

/*
 * @version V17.09
 */
public class Test{
    public static void main(String[] args) {
        HashMap<Integer, String> hm = init();

        System.out.println(+1);

        // 移出指定的键值对  ,这个键值对在集合中是存在的,所以删除成功 返回true
        System.out.println(hm.remove(5));
        System.out.println(hm);


    }

    public static HashMap<Integer, String> init() {
        HashMap<Integer, String> hm = new HashMap<Integer, String>();

        // 这里的键 在添加时是乱序的,然而在输出时 会有一个很有趣的现象
        // 要想知道这个现象背后的原因,就必须了解底层的代码实现
        // 所谓 玄之又玄,众妙之门
        hm.put(1, "北斗第一阳明贪狼太星君");
        hm.put(2, "北斗第二阴精巨门元星君");
        hm.put(5, "北斗第五丹元廉贞罡星君");
        hm.put(6, "北斗第六北极武曲纪星君");
        hm.put(7, "北斗第七天卫破军关星君");
        hm.put(3, "北斗第三福善禄存真星君");
        hm.put(4, "北斗第四玄冥文曲纽星君");
        hm.put(8, "北斗第八左辅洞明星君");
        hm.put(9, "北斗第九右弼隐光星君");

        return hm;
    }
}
