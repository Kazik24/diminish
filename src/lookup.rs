use std::collections::VecDeque;
use std::mem::{MaybeUninit, needs_drop, replace};
use std::fmt::Debug;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::convert::TryInto;

pub(crate) struct LookupTable<T,const SIZE: usize>{
    head: usize,
    count: usize,
    removes_until_rehash: usize,
    array: MaybeUninit<[T;SIZE]>,
    hashtable: [usize;HASH_SIZE], //SIZE is always power of two, so (SIZE*2)-1 is almost 2x larger (~0.5 load factor) and a prime
}

const HASH_SIZE: usize = 257;

unsafe fn get_array_elem<T,const N: usize>(array: &MaybeUninit<[T;N]>,index: usize)->&T{
    let ptr = array.as_ptr() as *const T;
    &*ptr.add(index)
}

impl<T: PartialEq + Debug + Hash,const SIZE: usize> LookupTable<T,SIZE> {

    const STATE_FREE: usize = usize::MAX;
    const STATE_REMOVED: usize = usize::MAX - 1;
    pub fn lookup_index_lin(&self,value: &T)->Option<usize>{
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



    fn lookup_hashtable_index_direct(&self,hash: u64,value: &T)->(isize,bool){
        let table = &self.hashtable;
        let mut index = (hash % (table.len() as u64)) as isize; //states.len() == points.len()
        let state = table[index as usize];
        if state == Self::STATE_FREE { return (index,false); }//free
        if state != Self::STATE_REMOVED && unsafe{ self.get_indexed_no_wrap(state as usize) } == value {
            return (index,true);
        }
        (index,false)
    }

    fn lookup_hashtable_index_indirect(&self,hash: u64,mut index: isize,value: &T)->Option<usize>{
        // see Knuth, p. 529
        let table = &self.hashtable;
        let probe = (1 + (hash % (table.len() as u64 - 2))) as isize;
        let loop_index = index; //loop over whole array until this index
        loop{
            index = index.wrapping_sub(probe);
            if index < 0 { index = index.wrapping_add(table.len() as _); }
            let state = table[index as usize];
            if state == Self::STATE_FREE { return None; } //first encountered Free breaks the loop
            if state != Self::STATE_REMOVED && unsafe{ self.get_indexed_no_wrap(state as usize) } == value {
                return Some(index as usize);
            }
            if index == loop_index { return None; } //looped over all elements without success
        }
    }
    pub fn lookup_hashtable_index(&self,value: &T)->Option<usize>{
        let hash = Self::hash_value(value); //hash has MSB always set to 0 (non negative)
        let (index,found) = self.lookup_hashtable_index_direct(hash,value);
        if found {
            return Some(index as usize);
        }
        self.lookup_hashtable_index_indirect(hash,index,value)
    }

    pub fn lookup_index(&self,value: &T)->Option<usize>{
        self.lookup_hashtable_index(value).map(|i|{
            Self::unwrap_index(self.head,self.hashtable[i])
        })
    }

    fn hash_value(value: &T)->u64{

        struct FastHasher{
            state: u8,
        }
        impl Hasher for FastHasher{
            fn finish(&self) -> u64 {
                self.state as _
            }
            fn write(&mut self, bytes: &[u8]) {
                for b in bytes {
                    self.state = self.state.wrapping_add(*b);
                    self.state ^= *b;
                }
            }
        }

        let mut hasher = FastHasher{state:0};
        value.hash(&mut hasher);
        hasher.finish()
    }

    fn split_mut(&mut self)->(&mut usize,&mut usize,&mut MaybeUninit<[T;SIZE]>,&mut [usize]){
        let array = self.hashtable.as_mut_ptr().cast::<usize>();
        (&mut self.head,
            &mut self.count,
            &mut self.array,
            &mut self.hashtable)
    }
    fn insert_hashtable_index(array: &MaybeUninit<[T;SIZE]>,hashtable: &mut [usize],value: &T,pushed_index: usize){
        let hash = Self::hash_value(value); //hash has MSB always set to 0 (non negative)
        let mut index = (hash % hashtable.len() as u64) as isize;
        let mut state = hashtable[index as usize];

        match state {
            Self::STATE_FREE => {
                hashtable[index as usize] = pushed_index; //reference inserted value
                return;
            }
            Self::STATE_REMOVED => {}
            // already stored
            state => {
                if unsafe{ get_array_elem(array,state) } == value {
                    panic!("Lookup table already contains element.");
                }
            }
        }

        let probe = (1 + (hash % (hashtable.len() as u64 - 2))) as isize;
        let mut first_removed = -1;
        let loop_index = index; //loop over whole array until this index

        //Look until FREE slot or we start to loop
        loop{
            // Identify first removed slot
            if state == Self::STATE_REMOVED && first_removed == -1 {first_removed = index;}
            index = index.wrapping_sub(probe);
            if index < 0 { index = index.wrapping_add(hashtable.len() as _); }
            state = hashtable[index as usize];
            // A Free slot stops the search


            if state == Self::STATE_FREE {
                let ret = if first_removed != -1 {
                    first_removed//insert in first removed pos
                }else{
                    index //insert in free pos
                };
                hashtable[ret as usize] = pushed_index; //reference inserted value
                return;
            } else if state != Self::STATE_REMOVED && unsafe{ get_array_elem(array,state) } == value {
                panic!("Lookup table already contains element.");
            }
            if index == loop_index {break;}
        }
        // We inspected all reachable slots and did not find a FREE one
        // If we found a REMOVED slot we return the first one found
        if first_removed != -1 {
            hashtable[first_removed as usize] = pushed_index;  // insert value
            return;
        }
        unreachable!("No free or removed slots available. Key set full?!!");
    }

    pub fn push(&mut self,value: T)->&T{
        let pushed_index = self.head.wrapping_add(1) & (SIZE - 1);
        //remove element from hashtable
        let ptr = unsafe{ self.ptr_mut().add(pushed_index) }; //pointer to place where value will reside
        if self.count == SIZE {
            unsafe{
                let prev = &*ptr;
                let hash = Self::hash_value(prev);
                let index = self.lookup_hashtable_index(prev).unwrap();

                self.hashtable[index] = Self::STATE_REMOVED;
                ptr.drop_in_place(); //value was removed from hashtable, now can be dropped
            }
            self.removes_until_rehash -= 1;
            if self.removes_until_rehash == 0 {
                self.removes_until_rehash = HASH_SIZE - SIZE - 1;
                self.rehash();
            }
        }else{//if queue is not full yet
            self.count += 1; //later code will insert actual value
        }


        let (_,_,array,hashtable) = self.split_mut();
        Self::insert_hashtable_index(array,hashtable,&value,pushed_index);
        self.head = pushed_index;
        unsafe{
            ptr.write(value);
            return &*ptr;
        }
    }


    fn rehash(&mut self){
        let mut new_hashtable = [usize::MAX;HASH_SIZE];
        let (_,_,array,hashtable) = self.split_mut();
        assert_eq!(new_hashtable.len(),hashtable.len());
        for state in hashtable.iter().copied() {
            if state < Self::STATE_REMOVED {
                let value = unsafe{ get_array_elem(array,state) };
                Self::insert_hashtable_index(array,&mut new_hashtable,value,state);
            }
        }
        hashtable.copy_from_slice(&new_hashtable);
    }
}

impl<T,const SIZE: usize> LookupTable<T,SIZE> {

    pub fn new()->Self{
        assert!(SIZE.is_power_of_two() && SIZE >= 4);
        Self{
            array: MaybeUninit::uninit(),
            count: 0,
            head: 0,
            removes_until_rehash: HASH_SIZE - SIZE - 1,
            hashtable: [usize::MAX;HASH_SIZE],
        }
    }

    fn wrap_index(head: usize,index: usize)->usize{
        head.wrapping_sub(index) & (SIZE - 1)
    }
    fn unwrap_index(head: usize,index: usize)->usize{
        head.wrapping_sub(index) & (SIZE - 1)
    }
    pub fn get(&self,index: usize)->Option<&T>{
        if index < self.count {
            let idx = Self::wrap_index(self.head,index);
            Some(unsafe{ self.get_indexed_no_wrap(idx) })
        } else {
            None
        }
    }
    unsafe fn get_indexed_no_wrap(&self,index: usize)->&T{
        &*self.ptr().add(index)
    }
    fn ptr(&self)->*const T{ self.array.as_ptr() as *const T}
    fn ptr_mut(&mut self)->*mut T{ self.array.as_mut_ptr() as *mut T}
    // #[inline]
    // fn hashtable(&self) ->&[usize]{
    //     let array = self.hashtable.as_ptr().cast::<usize>();
    //     unsafe{ from_raw_parts(array,SIZE*2-1) }
    // }
    // #[inline]
    // fn hashtable_mut(&mut self) ->&mut [usize]{
    //     let array = self.hashtable.as_mut_ptr().cast::<usize>();
    //     unsafe{ from_raw_parts_mut(array,SIZE*2-1) }
    // }

    fn push_internal(&mut self,value: T)->(&T,Option<T>){
        //advance head
        self.head = self.head.wrapping_add(1) & (SIZE - 1);
        if self.count == SIZE {
            //drop element at current position and overwrite it with new value
            unsafe{
                let ptr = &mut *self.ptr_mut().add(self.head);
                let prev = replace(ptr,value);
                (ptr,Some(prev))
            }
        }else{//if queue is not full yet
            self.count += 1;
            unsafe{ //only write value case we know that memory is uninit at this place
                let ptr = self.ptr_mut().add(self.head);
                ptr.write(value);
                (&*ptr,None)
            }
        }
    }
    pub fn push_lin(&mut self,value: T)->&T{
        self.push_internal(value).0
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
        debug_assert!(SIZE.is_power_of_two() && SIZE >= 4);
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
            if e == value {
                return Some(i);
            }
        }
        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::*;
    use rayon::iter::repeatn;
    use std::ops::Range;
    use rand::prelude::*;
    use std::cmp::{max, min};
    use std::time::Duration;


    #[test]
    fn test_todo(){


        println!("{}",LookupTable::<i32,128>::unwrap_index(1,2));

    }

    #[test]
    fn test_valid_drop(){

        static DROPPED: Mutex<Vec<i32>> = const_mutex(Vec::new());

        #[derive(Debug,PartialEq,Hash)]
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

    fn test_lookup_fill_drop<T: Hash + PartialEq,I,const S: usize,>(init_head: usize,values: I) where I: IntoIterator<Item=T>, T: Debug{
        let mut queue = LookupTable::<T,S>::new();
        queue.head = init_head & (S - 1);
        for v in values {
            queue.push(v);
        }
    }


    #[test]
    fn test_random_lookup(){


        const SIZE: usize = 128;

        #[derive(Eq,PartialEq,Hash,Debug,Copy,Clone)]
        struct Element(u32,u32,u32,u32,u32,u32,u32,u32);
        impl Element{
            fn new(val: u32)->Self{
                // Self(val,val.wrapping_add(456734252),val.wrapping_mul(4356),val.wrapping_div(3),val.wrapping_neg(),
                // val.wrapping_sub(554357579),val.wrapping_shr(14),val.wrapping_add(val.wrapping_mul(124)))
                Self(val,val,val,val,val,val,val,val)
            }
        }

        //todo, hash dziala szybciej na małych elementach a wolniej na dużych
        let result = benchmarking::bench_function_with_duration(Duration::from_secs(20),|m|m.measure(||{
            let mut rand = StdRng::seed_from_u64(thread_rng().next_u64());
            let mut table = LookupTableVec::<_,SIZE>::new();
            let mut queue = VecDeque::new();
            for elem in 1..10000 {
                let elem = Element::new(elem);
                table.push(elem);
                queue.push_front(elem);
                if queue.len() > SIZE {
                    queue.pop_back();
                }

                for _ in 0..min(queue.len() * 4,256) {
                    let index = rand.gen_range(0..queue.len());
                    let elem = &queue[index];
                    assert_eq!(table.lookup_index(elem),Some(index));
                }
            }
        })).unwrap();

        println!("Result: {:?}",result.total_elapsed()/result.times() as u32);

    }

}