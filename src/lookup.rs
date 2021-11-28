use std::collections::VecDeque;
use std::mem::{MaybeUninit, needs_drop};
use std::fmt::Debug;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub(crate) struct LookupTable<T,const SIZE: usize>{
    head: usize,
    count: usize,
    array: MaybeUninit<[T;SIZE]>,
    //hashtable: [[usize;2];SIZE], //SIZE is always power of two, so (SIZE*2)-1 is almost 2x larger and a prime
}




impl<T: PartialEq + Debug + Hash,const SIZE: usize> LookupTable<T,SIZE> {

    pub fn lookup_index(&self,value: &T)->Option<usize>{
        let mut index = self.head;
        let ptr = self.ptr();
        for i in 0..self.count {
            let val = unsafe { &*ptr.add(index) };
            if val == value {
                return Some(i);
            }
            index = Self::wrap_index(index,1);
        }
        None
    }

}

impl<T,const SIZE: usize> LookupTable<T,SIZE> {
    pub fn new()->Self{
        assert!(SIZE.is_power_of_two());
        Self{
            array: MaybeUninit::uninit(),
            count: 0,
            head: 0,
        }
    }

    fn wrap_index(head: usize,index: usize)->usize{
        head.wrapping_sub(index) & (SIZE - 1)
    }
    pub fn get(&self,index: usize)->Option<&T>{
        if index < self.count {
            let idx = Self::wrap_index(self.head,index);
            unsafe { Some(&*self.ptr().add(idx)) }
        } else {
            None
        }
    }
    fn ptr(&self)->*const T{ self.array.as_ptr() as *const T}
    fn ptr_mut(&mut self)->*mut T{ self.array.as_mut_ptr() as *mut T}
    // fn hashtable(&self) ->&[usize]{
    //     let array = self.hashtable.as_ptr().cast::<usize>();
    //     unsafe{ from_raw_parts(array,SIZE*2-1) }
    // }
    // fn hashtable_mut(&mut self) ->&mut [usize]{
    //     let array = self.hashtable.as_mut_ptr().cast::<usize>();
    //     unsafe{ from_raw_parts_mut(array,SIZE*2-1) }
    // }
    pub fn push(&mut self,value: T){
        //advance head
        self.head = self.head.wrapping_add(1) & (SIZE - 1);
        if self.count == SIZE {
            //drop element at current position and overwrite it with new value
            unsafe{
                *self.ptr_mut().add(self.head) = value;
            }
        }else{//if queue is not full yet
            self.count += 1;
            unsafe{ //only write value case we know that memory is uninit at this place
                self.ptr_mut().add(self.head).write(value);
            }
        }
    }
    pub fn is_empty(&self)->bool{ self.count == 0 }
    fn ring_slices(buf: &mut [T;SIZE], head: usize, count: usize) -> (&mut [T], &mut [T]) {
        let (left,right) = buf.split_at_mut(head + 1);
        if count <= head {//all on left side oh head (head always moves from left to right)
            let (_,full) = left.split_at_mut(left.len() - count);
            (full,&mut [])
        }else{
            let rest = SIZE - count;
            let (_,right) = right.split_at_mut(rest);
            (left,right)
        }
    }
}

impl<T,const SIZE: usize> Drop for LookupTable<T,SIZE>{
    fn drop(&mut self) {
        if needs_drop::<T>() {
            struct Dropper<'a, T>(&'a mut [T]);

            impl<'a, T> Drop for Dropper<'a, T> {
                fn drop(&mut self) {
                    unsafe { core::ptr::drop_in_place(self.0); }
                }
            }

            unsafe {
                let (front, back) = Self::ring_slices(self.array.assume_init_mut(),self.head,self.count);//todo
                let _back_dropper = Dropper(back);
                // use drop for [T]
                core::ptr::drop_in_place(front);
            }
        }
    }
}

pub(crate) struct LookupTableVec<T,const SIZE: usize>{
    lookup: VecDeque<T>,
}
impl<T: PartialEq,const SIZE: usize> LookupTableVec<T,SIZE> {
    pub fn new()->Self{
        debug_assert!(SIZE.is_power_of_two());
        Self{
            lookup: VecDeque::with_capacity(SIZE),
        }
    }
    pub fn get(&self,index: usize)->Option<&T>{ self.lookup.get(index) }
    pub fn push(&mut self,value: T){
        if self.lookup.len() >= SIZE {
            self.lookup.pop_back();
        }
        self.lookup.push_front(value);
    }
    pub fn is_empty(&self)->bool{ self.lookup.is_empty() }
    pub fn lookup_index(&self,value: &T)->Option<usize> where T: Debug{
        for (i,e) in self.lookup.iter().enumerate() {
            println!("Scan val at: {} {:?}",i,e);
            if e == value {
                println!("Found at: {}",i);
                return Some(i);
            }
        }
        println!("Not found");
        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::*;
    use rayon::iter::repeatn;
    use std::ops::Range;

    #[test]
    fn test_valid_drop(){

        static DROPPED: Mutex<Vec<i32>> = const_mutex(Vec::new());

        #[derive(Debug)]
        struct Droppable(i32);
        impl Drop for Droppable{
            fn drop(&mut self) {
                DROPPED.lock().push(self.0);
            }
        }

        //test unittest
        drop(Droppable(10));
        assert_eq!(DROPPED.lock().pop(),Some(10));
        assert!(DROPPED.lock().is_empty());

        //test partial fill
        fn test_fill<const S: usize>(heads: usize,range: Range<usize>){
            for i in range {
                for head in 0..heads {
                    test_lookup_fill_drop::<_,_,S>(head,(0..i).map(|v|Droppable((v+1) as _)));

                    let mut vec = vec![false;i];

                    for val in DROPPED.lock().drain(..){
                        vec[val as usize - 1] = true;
                    }
                    if !vec.iter().all(|v|*v) {
                        println!("{}: {:?}",i,vec.iter().enumerate().filter(|(_,v)|!**v).collect::<Vec<_>>());
                    }
                    assert!(vec.iter().all(|v|*v));
                }
            }
        }

        //test cases
        test_fill::<8>(8,0..8);
        test_fill::<8>(8,8..64);
        test_fill::<8>(8,64..128);
        test_fill::<32>(33,0..8);
        test_fill::<32>(33,8..64);
        test_fill::<32>(33,64..128);
        test_fill::<32>(33,128..256);
        test_fill::<128>(128,0..128);
        test_fill::<128>(128,128..256);
    }

    fn test_lookup_fill_drop<T,I,const S: usize,>(init_head: usize,values: I) where I: IntoIterator<Item=T>, T: Debug{
        let mut queue = LookupTable::<T,S>::new();
        queue.head = init_head & (S - 1);
        for v in values {
            queue.push(v);
        }
    }

}