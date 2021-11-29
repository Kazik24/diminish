mod bit;
mod types;
mod lookup;

pub use bit::*;

use std::io::{Write, Result, Read, Error, ErrorKind};
use std::hash::Hash;
use std::collections::VecDeque;
use std::mem::replace;
use crate::BitWriter;
use crate::lookup::{LookupTable, LookupTableVec};
use std::fmt::Debug;
use std::borrow::{Cow, Borrow};


pub trait Serializable: PartialEq + Hash + Debug{ //todo remove debug
    fn convert_to_bits<W: Write>(&self,write: &mut BitWriter<'_,W>)->Result<()>;
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_,R>)->Result<Self> where Self: Sized;
}

impl Serializable for bool{
    fn convert_to_bits<W: Write>(&self, write: &mut BitWriter<'_, W>) -> Result<()> { write.write_bit(*self) }
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_, R>) -> Result<Self> { read.read_bit() }
}
impl Serializable for u8{
    fn convert_to_bits<W: Write>(&self, write: &mut BitWriter<'_, W>) -> Result<()> { write.write_u8(*self) }
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_, R>) -> Result<Self> { read.read_u8() }
}
impl Serializable for u16{
    fn convert_to_bits<W: Write>(&self, write: &mut BitWriter<'_, W>) -> Result<()> { write.write_u16(*self) }
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_, R>) -> Result<Self> { read.read_u16() }
}
impl Serializable for u32{
    fn convert_to_bits<W: Write>(&self, write: &mut BitWriter<'_, W>) -> Result<()> { write.write_u32(*self) }
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_, R>) -> Result<Self> { read.read_u32() }
}
impl Serializable for i32{
    fn convert_to_bits<W: Write>(&self, write: &mut BitWriter<'_, W>) -> Result<()> { write.write_u32(*self as _) }
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_, R>) -> Result<Self> { read.read_u32().map(|v|v as _) }
}





const MAX_LOOKUP: usize = 128;

pub struct Encoder<W: Write,V> { //todo remove debug
    write: W,
    run_length: u32,
    last_lookup_index: usize,
    lookup: LookupTable<V,MAX_LOOKUP>,
}


impl<W: Write,V: Serializable> Encoder<W,V> {

    pub fn new(write: W)->Self{
        Self{
            write,
            run_length: 0,
            last_lookup_index: 0,
            lookup: LookupTable::new(),
        }
    }
    pub fn get_ref(&self)->&W{&self.write}
    pub fn get_mut(&mut self)->&mut W{&mut self.write}
    pub fn into_inner(self)->W{self.write}

    pub fn write_from(&mut self,iter: impl IntoIterator<Item=V>)->Result<usize>{
        let mut count = 0;
        for elem in iter {
            self.write_value(elem)?;
            count += 1;
        }
        Ok(count)
    }
    pub fn write_chunk_ref(&mut self,value: &V,count: u32)->Result<()> where V: Clone{
        self.write_value_internal(value,Clone::clone,count)
    }
    pub fn write_chunk(&mut self,value: V,count: u32)->Result<()>{
        self.write_value_internal(value,|v|v,count)
    }
    pub fn write_value_ref(&mut self,value: &V)->Result<()> where V: Clone{
        self.write_value_internal(value,Clone::clone,1)
    }
    pub fn write_value(&mut self,value: V)->Result<()>{
        self.write_value_internal(value,|v|v,1)
    }

    pub fn write_value_internal<T>(&mut self,value: T,clone: impl FnOnce(T)->V,count: u32)->Result<()> where T: Borrow<V>{
        if count == 0 { return Ok(()); }
        match self.lookup.get(self.last_lookup_index) {
            Some(val) => {
                if val == value.borrow() {
                    self.run_length += count;
                    if self.run_length >= 0b1111_1111_1111_1111_1111 + 0b1_1111_1111_1111 + 0b0011_1111 {
                        self.write_run_length_marker()?;
                    }
                } else {
                    if self.run_length != 0 {
                        //write run length marker
                        self.write_run_length_marker()?;
                    }
                    self.run_length = count - 1;
                    //perform lookup of index
                    match self.lookup.lookup_index(value.borrow()) {
                        Some(idx) => {
                            //prefix 0b0
                            self.write.write_all(&[idx as u8])?; // 0b0iiiiiii
                            self.last_lookup_index = idx;
                        }
                        None => {
                            Self::write_to_stream(&mut self.write,value.borrow());
                            self.lookup.push(clone(value));
                            self.last_lookup_index = 0;
                        }
                    }
                }
            }
            None => { //init
                self.run_length = count - 1; //not needed cause obj init?
                self.last_lookup_index = 0;//not needed cause obj init?
                Self::write_to_stream(&mut self.write,value.borrow())?;
                self.lookup.push(clone(value));
            }
        }
        Ok(())
    }

    pub fn flush(&mut self)->Result<()>{
        while self.run_length != 0 {
            self.write_run_length_marker()?;
        }
        Ok(())
    }


    fn write_run_length_marker(&mut self)->Result<()>{
        let mut size = self.run_length - 1;
        if size <= 0b0011_1111 { //prefix 0b10
            self.run_length = 0;
            self.write.write_all(&[(0b1000_0000 | size) as u8])?;
        }else if size <= 0b1_1111_1111_1111 + 0b0011_1111{ //prefix 0b110
            self.run_length = 0;//todo subtract previous offset from size
            size -= 0b0011_1111;
            let arr = &((0b1100_0000_0000_0000 | size) as u16).to_be_bytes();
            self.write.write_all(arr)?;
        }else if size <= 0b1111_1111_1111_1111_1111 + 0b1_1111_1111_1111 + 0b0011_1111 { //prefix 0b1110
            self.run_length = 0;//todo subtract previous offset from size
            size -= 0b1_1111_1111_1111 + 0b0011_1111;
            let arr = &((0b1110_0000_0000_0000_0000_0000 | size) as u32).to_be_bytes()[1..];
            self.write.write_all(arr)?;
        }else{ //prefix 0b1110 maximum value
            //todo might be buggy with very high numbers and u64
            self.run_length -= 0b1111_1111_1111_1111_1111 + 0b1_1111_1111_1111 + 0b0011_1111;
            self.write.write_all(&[0b1110_1111,0xff,0xff])?;
        }
        Ok(())
    }

    fn write_to_stream(stream: &mut W,value: &V)->Result<()>{
        //prefix 0b1111
        let mut wr = BitWriter{
            inner: stream,
            current: 0b11110000,
            length: 4,
        };
        value.convert_to_bits(&mut wr)?;
        wr.flush()
    }

}

pub struct Decoder<R: Read,V: Serializable>{
    read: R,
    run_length: u32,
    last_lookup_index: usize,
    lookup: LookupTable<V,MAX_LOOKUP>,
}

impl<R: Read,V: Serializable> Decoder<R,V> {

    pub fn new(read: R)->Self{
        Self{
            read,
            last_lookup_index: 0,
            run_length: 0,
            lookup: LookupTable::new(),
        }
    }
    pub fn get_ref(&self)->&R{&self.read}
    pub fn get_mut(&mut self)->&mut R{&mut self.read}
    pub fn into_inner(self)->R{self.read}

    fn read_token<'a>(read: &mut R,lookup: &'a mut LookupTable<V,MAX_LOOKUP>,last_lookup_index: &mut usize)->Result<(&'a V,u32)>{
        let mut byte = 0;
        read.read_exact(std::slice::from_mut(&mut byte))?;
        if byte <= 0b0111_1111 {
            let index = byte as usize;
            match lookup.get(index) {
                Some(value) => {
                    *last_lookup_index = index;
                    Ok((value,0))
                }
                None => {
                    Err(Error::new(ErrorKind::InvalidData,"Lookup index is out of bounds for current table size."))
                }
            }
        }else if byte <= 0b1011_1111 {
            if lookup.is_empty() {
                return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
            }
            let update_run = (byte & 0x3f) as _;
            Ok((lookup.get(*last_lookup_index).unwrap(),update_run))
        }else if byte <= 0b1101_1111 {
            if lookup.is_empty() {
                return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
            }
            let mut update_run = ((byte & 0x1f) as u32) << 8;
            let mut byte = 0;
            read.read_exact(std::slice::from_mut(&mut byte))?;
            update_run |= byte as u32;
            update_run += 0b0011_1111;
            Ok((lookup.get(*last_lookup_index).unwrap(),update_run))
        }else if byte <= 0b1110_1111 {
            if lookup.is_empty() {
                return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
            }
            let mut update_run = ((byte & 0xf) as u32) << 16;
            let mut lower_val = [0;2];
            read.read_exact(&mut lower_val)?;
            update_run |= u16::from_be_bytes(lower_val) as u32;
            update_run += 0b1_1111_1111_1111 + 0b0011_1111;
            Ok((lookup.get(*last_lookup_index).unwrap(),update_run))
        }else{
            let mut read = BitReader{
                inner: read,
                current: byte,
                length: 4,
            };
            let value = V::convert_from_bits(&mut read)?;
            *last_lookup_index = 0; //always valid after this operation
            let value = lookup.push(value);
            Ok((value,0))
        }
    }

    pub fn read_chunk(&mut self)->Result<(&V,u32)>{
        if self.run_length != 0 {
            let len = self.run_length;
            self.run_length = 0;
            return Ok((self.lookup.get(self.last_lookup_index).unwrap(),len)); //todo error instead of panic
        }
        let (val,rl) = Self::read_token(&mut self.read,&mut self.lookup,&mut self.last_lookup_index)?;
        self.run_length = rl;
        if rl == 0 { Ok((val,1)) }
        else { Ok((val,rl)) }
    }
    pub fn read_value_ref(&mut self)->Result<&V>{
        if self.run_length != 0 {
            self.run_length -= 1;
            return Ok(self.lookup.get(self.last_lookup_index).unwrap()); //todo error instead of panic
        }
        let (val,rl) = Self::read_token(&mut self.read,&mut self.lookup,&mut self.last_lookup_index)?;
        self.run_length = rl;
        Ok(val)
    }
    pub fn read_value(&mut self)->Result<V> where V: Clone{
        self.read_value_ref().map(Clone::clone)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::fmt::Debug;
    use rand::rngs::StdRng;
    use rand::{SeedableRng, Rng};
    use rand::prelude::*;
    use rayon::iter::*;
    use rayon::prelude::*;


    fn run_test_encode_decode<T: Serializable + Clone + Debug>(data: &[T])->usize{
        let mut write = Encoder::new(Vec::new());
        for val in data.iter().cloned() {
            write.write_value(val).unwrap();
        }
        write.flush().unwrap();

        // println!("********** Size: {}",write.get_ref().len());
        // for b in write.get_ref() {
        //     println!("{:#010b}",b);
        // }

        let bytes = write.into_inner();
        let len = bytes.len();
        let mut read = Decoder::new(Cursor::new(bytes));

        let mut result: Vec<T> = Vec::with_capacity(data.len() + 10);
        for _ in 0..data.len() {
            result.push(read.read_value().unwrap());
        }
        assert_eq!(result.as_slice(),data,"Values expected: {} result: {}",data.len(),result.len());
        len
    }

    #[test]
    fn test_complement(){
        println!("Size: {}",run_test_encode_decode(&[1,1,2,1,3,1,4,4,1,11111,11,11,11,11,11,11,11,1]));
        println!("Size: {}",run_test_encode_decode(&vec![123;2000000]));
        println!("Size: {}",run_test_encode_decode(&vec![123;200000]));
        println!("Size: {}",run_test_encode_decode(&vec![123;200]));
        println!("Size: {}",run_test_encode_decode(&vec![123;20]));
    }

    fn run_test_distinct(min_distinct: usize,max_distinct: usize,iters: usize,max_elem: usize,seed: u64){
        let mut rand = &mut StdRng::seed_from_u64(seed);


        let mut vec = Vec::new();
        let mut fixed_pool = Vec::new();
        for _ in 0..iters {
            fixed_pool.clear();
            for _ in 0..rand.gen_range(min_distinct..=max_distinct) {
                fixed_pool.push(rand.gen::<u32>());
            }
            for _ in 1..rand.gen_range(1..=max_elem) {
                vec.push(fixed_pool.choose(rand).copied().unwrap());
            }
            run_test_encode_decode(&vec);
        }
    }

    #[test]
    fn test_low_distinct(){
        run_test_distinct(1,MAX_LOOKUP,100,1000,123456);
        run_test_distinct(1,MAX_LOOKUP,20,1000,987654321);
        run_test_distinct(1,MAX_LOOKUP,20,2000,42069);
        run_test_distinct(1,MAX_LOOKUP,20,3000,6969420420);
    }
    #[test]
    fn test_high_distinct(){
        run_test_distinct(MAX_LOOKUP-40,MAX_LOOKUP+1,100,1000,123456);
        run_test_distinct(MAX_LOOKUP-30,MAX_LOOKUP+10,20,1000,987654321);
        run_test_distinct(MAX_LOOKUP-20,MAX_LOOKUP+100,20,2000,42069);
        run_test_distinct(MAX_LOOKUP-10,MAX_LOOKUP+1000,20,3000,6969420420);
    }
    #[test]
    fn test_basic_write() {
        fn test_case<T: PartialEq + Debug + Serializable + Clone>(data: impl IntoIterator<Item=T>,expect: Vec<u8>){
            let mut write = Encoder::new(Vec::new());
            let vec = data.into_iter().collect::<Vec<_>>();
            let len = vec.len();
            assert_eq!(write.write_from(vec).unwrap(),len);
            assert!(write.flush().is_ok());
            assert_eq!(write.into_inner(),expect);
        }

        test_case([1i32,1,2,1,3,1,4,4,1,1,1],
                  vec![240, 0, 0, 0, 16, 128, 240, 0, 0, 0, 32, 1, 240, 0, 0, 0, 48, 2, 240, 0, 0, 0, 64, 128, 3, 129])

    }

    #[test]
    fn test_basic_read() {
        fn test_case<T: PartialEq + Debug + Serializable + Clone>(data: Vec<u8>,expect: Vec<T>){
            let expect_read = data.len();
            let mut read = Decoder::new(Cursor::new(data));

            let mut result: Vec<T> = Vec::new();
            for _ in 0..expect.len() {
                result.push(read.read_value().unwrap());
            }
            assert_eq!(read.get_ref().position(),expect_read as u64);
            assert_eq!(read.read_value().unwrap_err().kind(),ErrorKind::UnexpectedEof);
            assert_eq!(result,expect);
        }


        test_case(vec![240, 0, 0, 0, 16, 128, 240, 0, 0, 0, 32, 1, 240, 0, 0, 0, 48, 2, 240, 0, 0, 0, 64, 128, 3, 129],
                  vec![1i32,1,2,1,3,1,4,4,1,1,1]);

    }
    #[test]
    fn test_read_random_data(){
        let mut rand = &mut StdRng::seed_from_u64(123456789);
        let mut vec = vec![0u8;1000000];

        for _ in 0..1000 {
            let range = &mut vec[0..rand.gen_range(1..10000)];
            rand.fill_bytes(range);

            let mut read = Decoder::<_,i32>::new(Cursor::new(range));

            while let Ok(val) = read.read_value() {}
        }
    }
    //#[test]
    fn test_read_any_data_no_panic(){
        let it: rayon::range_inclusive::Iter<u32> = (0..=u32::MAX).into_par_iter();
        it.for_each(|v|{
            let mut read = Decoder::<_,u8>::new(Cursor::new(v.to_be_bytes()));

            read.read_chunk();
        });


        // for v in 0..=u32::MAX {
        //     let array = v.to_be_bytes();
        //
        //     // let mut read = Decoder::<_,u8>::new(Cursor::new(array));
        //     //
        //     // while let Ok(val) = read.read_value() {}
        // }
    }
}

