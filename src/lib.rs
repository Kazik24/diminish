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


pub trait Serializable: PartialEq + Hash + Debug{ //todo remove debug
    fn convert_to_bits<W: Write>(&self,write: &mut BitWriter<'_,W>)->Result<()>;
    fn convert_from_bits<R: Read>(read: &mut BitReader<'_,R>)->Result<Self> where Self: Sized + Clone;
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
    pub fn write_value(&mut self,value: V)->Result<()>{
        match self.lookup.get(self.last_lookup_index) {
            Some(val) => {
                if val == &value {
                    self.run_length += 1;
                    if self.run_length >= 0b1111_1111_1111_1111_1111 + 0b1_1111_1111_1111 + 0b0011_1111 {
                        self.write_run_length_marker()?;
                    }
                } else {
                    if self.run_length != 0 {
                        //write run length marker
                        self.write_run_length_marker()?;
                    }
                    self.run_length = 0;
                    //perform lookup of index
                    match self.lookup.lookup_index(&value) {
                        Some(idx) => {
                            //prefix 0b0
                            self.write.write_all(&[idx as u8])?; // 0b0iiiiiii
                            self.last_lookup_index = idx;
                        }
                        None => {
                            Self::write_to_stream(&mut self.write,&value);
                            self.lookup.push(value);
                            // if self.lookup.len() >= MAX_LOOKUP {
                            //     self.lookup.pop_back();
                            // }
                            // self.lookup.push_front(value);
                            self.last_lookup_index = 0;
                        }
                    }
                }
            }
            None => { //init
                self.run_length = 0; //not needed cause obj init?
                self.last_lookup_index = 0;//not needed cause obj init?
                Self::write_to_stream(&mut self.write,&value)?;
                self.lookup.push(value);
            }
        }
        Ok(())
    }

    pub fn flush(&mut self)->Result<()>{
        if self.run_length != 0 {
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

pub struct Decoder<R: Read,V: Serializable + Clone>{
    read: R,
    run_length: u32,
    last_lookup_index: usize,
    lookup: LookupTable<V,MAX_LOOKUP>,
}

impl<R: Read,V: Serializable + Clone> Decoder<R,V> {

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
    pub fn read_chunk(&mut self)->Result<(V,u32)>{
        if self.run_length != 0 {
            let len = self.run_length;
            self.run_length = 0;
            return Ok((self.lookup.get(self.last_lookup_index).unwrap().clone(),len)); //todo error instead of panic
        }
        let (val,rl) = self.read_token()?;
        self.run_length = rl;
        if rl == 0 { Ok((val,1)) }
        else { Ok((val,rl)) }
    }


    fn read_token(&mut self)->Result<(V,u32)>{
        let mut byte = 0;
        self.read.read_exact(std::slice::from_mut(&mut byte))?;
        match byte {
            0..=0b0111_1111 => {
                let index = byte as usize;
                match self.lookup.get(index) {
                    Some(value) => {
                        self.last_lookup_index = index;
                        Ok((value.clone(),0))
                    }
                    None => {
                        Err(Error::new(ErrorKind::InvalidData,"Lookup index is out of bounds for current table size."))
                    }
                }
            }
            0b1000_0000..=0b1011_1111 => {
                if self.lookup.is_empty() {
                    return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
                }
                let update_run = (byte & 0x3f) as _;
                Ok((self.lookup.get(self.last_lookup_index).unwrap().clone(),update_run))
            }
            0b1100_0000..=0b1101_1111 => {
                if self.lookup.is_empty() {
                    return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
                }
                let mut update_run = ((byte & 0x1f) as u32) << 8;
                let mut byte = 0;
                self.read.read_exact(std::slice::from_mut(&mut byte))?;
                update_run |= byte as u32;
                update_run += 0b0011_1111;
                Ok((self.lookup.get(self.last_lookup_index).unwrap().clone(),update_run))
            }
            0b1110_0000..=0b1110_1111 => {
                if self.lookup.is_empty() {
                    return Err(Error::new(ErrorKind::InvalidData,"Cannot reference value from empty lookup table."))
                }
                let mut update_run = ((byte & 0xf) as u32) << 16;
                let mut byte = 0;
                self.read.read_exact(std::slice::from_mut(&mut byte))?;
                update_run |= (byte as u32) << 8;
                self.read.read_exact(std::slice::from_mut(&mut byte))?;
                update_run |= byte as u32;
                update_run += 0b1_1111_1111_1111 + 0b0011_1111;
                Ok((self.lookup.get(self.last_lookup_index).unwrap().clone(),update_run))
            }
            0b1111_0000..=0b1111_1111 => {
                let mut read = BitReader{
                    inner: &mut self.read,
                    current: byte,
                    length: 4,
                };
                let value = V::convert_from_bits(&mut read)?;
                self.last_lookup_index = 0; //always valid after this operation
                self.lookup.push(value.clone());
                Ok((value,0))
            }
        }
    }
    pub fn read_value(&mut self)->Result<V>{
        if self.run_length != 0 {
            self.run_length -= 1;
            return Ok(self.lookup.get(self.last_lookup_index).unwrap().clone()); //todo error instead of panic
        }
        let (val,rl) = self.read_token()?;
        self.run_length = rl;
        Ok(val)
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

        let mut write = Encoder::new(Vec::new());

        write.write_value(1);
        write.write_value(1);
        write.write_value(2);
        write.write_value(1);
        write.write_value(3);
        write.write_value(1);
        write.write_value(4);
        write.write_value(4);
        write.write_value(1);
        write.write_value(1);
        write.write_value(1);
        write.flush();

        // println!("********** Size: {}",write.get_ref().len());
        // for b in write.get_ref() {
        //     println!("{:08b}",b);
        // }
        assert_eq!(write.into_inner(),
                   vec![240, 0, 0, 0, 16, 128, 240, 0, 0, 0, 32, 1, 240, 0, 0, 0, 48, 2, 240, 0, 0, 0, 64, 128, 3, 129])


    }

    #[test]
    fn test_basic_read() {

        let data = vec![240,23,222,111,212,000,1,2,4,79,0,0];
        let mut read = Decoder::new(Cursor::new(data));

        let mut result: Vec<i32> = Vec::new();
        while let Ok(v) = read.read_value() {
            result.push(v);
        }

        println!("{:?}",result);


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
    #[test]
    fn test_read_any_data_no_panic(){
        // let it: rayon::range_inclusive::Iter<u32> = (0..=u32::MAX).into_par_iter();
        // it.for_each(|v|{
        //     let array = v.to_be_bytes();
        //     let mut read = Decoder::<_,u8>::new(Cursor::new(array));
        //
        //     read.read_value();
        // });


        // for v in 0..=u32::MAX {
        //     let array = v.to_be_bytes();
        //
        //     // let mut read = Decoder::<_,u8>::new(Cursor::new(array));
        //     //
        //     // while let Ok(val) = read.read_value() {}
        // }
    }
}

