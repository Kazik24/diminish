use std::io::*;
use std::mem::{replace, size_of};
use std::slice::from_mut;
use std::convert::TryInto;
use bitstream_io::{BitRead, Numeric, SignedNumeric};

pub struct BitWriter<'a,W: Write>{
    pub(crate) inner: &'a mut W,
    pub(crate) current: u8,
    pub(crate) length: u8,
}

pub struct BitReader<'a,R: Read>{
    pub(crate) inner: &'a mut R,
    pub(crate) current: u8,
    pub(crate) length: u8,
}


impl<'a,W: Write> BitWriter<'a,W>{



    pub fn write_u32(&mut self,value: u32)->Result<()>{
        self.write_u16(value.wrapping_shr(16) as _)?;
        self.write_u16((value & 0xffff) as _)
    }
    pub fn write_u16(&mut self,mut value: u16)->Result<()>{
        self.write_u8(value.wrapping_shr(8) as _)?;
        self.write_u8((value & 0xff) as _)
    }
    pub fn write_u8(&mut self,mut value: u8)->Result<()>{
        for _ in 0..8 {
            self.write_bit(value & 0x80 != 0)?;
            value = value.wrapping_shl(1);
        }
        Ok(())
    }
    pub fn write_bit(&mut self,value: bool)->Result<()>{
        if value {
            self.current |= 1 << 7 - self.length;
        }
        if self.length == 7 {
            self.length = 0;
            self.inner.write_all(&[replace(&mut self.current,0)])
        }else{
            self.length += 1;
            Ok(())
        }
    }

    #[inline]
    pub fn bits_until_align(&self)->usize{ (7 - self.length) as _ }
    #[inline]
    pub fn is_aligned(&self)->bool { self.length == 0 }
    #[inline]
    pub fn aligned(&mut self)->Option<&mut W>{
        if self.is_aligned() { Some(self.inner) }
        else { None }
    }

    pub(crate) fn flush(mut self)->Result<()>{
        if self.length != 0 {
            self.inner.write_all(&[self.current])?;
        }
        Ok(())
    }
}


impl<'a,R: Read> Read for BitReader<'a,R>{
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        if buf.is_empty() { return Ok(0); }
        if self.length == 0 {
            self.inner.read(buf)
        }else{
            let len = self.inner.read(buf)?;
            let last = *buf.last().unwrap();//cannot fail
            for i in (1..len).rev() {
                let mut value = u16::from_be_bytes(buf[(i-1)..=i].try_into().unwrap()); //cannot fail
                buf[i] = value.wrapping_shr(self.length as _) as u8;
            }
            buf[0] = u16::from_be_bytes([self.current,buf[0]]).wrapping_shr(self.length as _) as u8;
            self.current = last;
            Ok(len)
        }
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()>{
        if self.is_aligned() {
            self.inner.read_exact(buf)
        }else{
            if buf.is_empty() { return Ok(()); }
            self.inner.read_exact(buf)?;
            let last = *buf.last().unwrap();//cannot fail
            for i in (1..buf.len()).rev() {
                let mut value = u16::from_be_bytes(buf[(i-1)..=i].try_into().unwrap()); //cannot fail
                buf[i] = value.wrapping_shr(self.length as _) as u8;
            }
            buf[0] = u16::from_be_bytes([self.current,buf[0]]).wrapping_shr(self.length as _) as u8;
            self.current = last;
            Ok(())
        }

    }
}

macro_rules! def_read_func{
    ($func_name:ident,$func_type:ty,$big_type:ty) => {
        pub fn $func_name(&mut self)->Result<$func_type>{
            if self.is_aligned() {
                let mut v = [0;size_of::<$func_type>()];
                self.inner.read_exact(&mut v).map(|_|<$func_type>::from_be_bytes(v))
            }else{
                let mut v = [0;size_of::<$func_type>()*2];
                v[size_of::<$func_type>()-1] = self.current;
                self.inner.read_exact(&mut v[size_of::<$func_type>()..])?;
                let result = <$big_type>::from_be_bytes(v).wrapping_shr(self.length as _) as $func_type;
                self.current = v[(size_of::<$func_type>()*2) - 1];
                Ok(result)
            }
        }
    };
    ($func_name:ident,$func_type:ty) => {
        pub fn $func_name(&mut self)->Result<$func_type>{
            let mut data = [0;size_of::<$func_type>()];
            self.read_exact(&mut data).map(|_|<$func_type>::from_be_bytes(data))
        }
    };
}

impl<'a,R: Read> BitReader<'a,R>{

    pub fn new(read: &'a mut R)->Self{
        Self{ inner: read, current: 0, length: 0 }
    }


    def_read_func!(read_u128,u128);
    def_read_func!(read_u64,u64);
    def_read_func!(read_u32,u32,u64);
    def_read_func!(read_u16,u16,u32);
    def_read_func!(read_u8,u8,u16);

    #[inline]
    pub fn bits_until_align(&self)->usize{ self.length as _ }
    #[inline]
    pub fn align_to_byte(&mut self)->(u8,usize){ //returns byte and count, count == 0 means that no skip was needed
        let result = match self.length {
            0 => (0,0),
            val => {
                let mask = ((1u32 << val) - 1) as u8;
                (self.current & mask,val as _)
            }
        };
        self.length = 0;
        result
    }
    #[inline]
    pub fn is_aligned(&self)->bool { self.length == 0 }
    #[inline]
    pub fn aligned(&mut self)->Option<&mut R>{
        if self.is_aligned() { Some(self.inner) }
        else { None }
    }

    pub fn read_bit(&mut self)->Result<bool>{
        if self.is_aligned() {
            self.inner.read_exact(from_mut(&mut self.current))?;
            self.length = 7;
        }else{
            self.length -= 1;
        }
        Ok(self.current & (1 << self.length) != 0)
    }
}


impl<'a,R: Read> BitRead for BitReader<'a,R>{
    fn read_bit(&mut self) -> Result<bool> {
        todo!()
    }

    fn read<U>(&mut self, bits: u32) -> Result<U> where U: Numeric {
        todo!()
    }

    fn read_signed<S>(&mut self, bits: u32) -> Result<S> where S: SignedNumeric {
        todo!()
    }

    fn skip(&mut self, bits: u32) -> Result<()> {
        todo!()
    }

    fn byte_aligned(&self) -> bool {
        todo!()
    }

    fn byte_align(&mut self) {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::{Debug, Binary, Display};
    use std::iter::*;
    use rand::prelude::*;
    use std::borrow::{Borrow, BorrowMut};


    fn bit_writer(vec: &mut Vec<u8>) -> BitWriter<Vec<u8>> { bit_writer_init(vec,&[]) }
    fn bit_writer_init<'a>(vec: &'a mut Vec<u8>,array: &[bool]) -> BitWriter<'a,Vec<u8>> {
        assert!(array.len() <= 8);
        let mut current = 0;
        for (i,v) in array.iter().copied().enumerate() {
            if v {
                current |= 1 << (7 - i);
            }
        }
        BitWriter {
            inner: vec,
            current,
            length: array.len() as _,
        }
    }

    #[test]
    fn test_bit_write() {
        let mut w = Vec::new();
        let mut bit = bit_writer(&mut w);

        bit.write_bit(true);
        bit.write_bit(false);
        bit.write_bit(true);
        bit.flush();
        assert_eq!(w, vec![0b1010_0000]);
    }

    fn assert_read<T: PartialEq + Debug>(data: impl AsRef<[u8]>,offset: u8,expect: impl Borrow<T>,func: impl FnOnce(BitReader<'_,Cursor<&'_ [u8]>>)->T){
        assert!(offset <= 8);
        let off = offset % 8;
        let data = data.as_ref();
        let mut v = Cursor::new(data);
        if offset != 0 {
            let count = (offset / 8 + if offset % 8 == 0 { 0 } else { 1 }) as _;
            v.set_position(count);
        }
        let result = func(BitReader {
            inner: &mut v,
            current: if offset == 0 { 0 } else {data[0]},//not exacly needed, just data[0] is ok but it improves test cases
            length: if off == 0 { 0 } else {8 - off},
        });
        assert_eq!(&result,expect.borrow());
    }

    #[test]
    fn test_read_u8(){
        fn assert_read_v(data: impl AsRef<[u8]>, off: u8, expect: u8){
            assert_read(data,off,expect,|mut r|r.read_u8().unwrap());
        }
        fn assert_read_arr(data: impl AsRef<[u8]>, off: u8, expect: u8){
            assert_read(data,off,expect,|mut r|{
                let mut val = 0;
                r.read_exact(from_mut(&mut val)).map(|_|val).unwrap()
            });
        }
        assert_read_v([0b10101010,0b01010101], 0, 0b10101010);
        assert_read_v([0b10101010,0b01010101], 1, 0b01010100);
        assert_read_v([0b10101010,0b01010101], 2, 0b10101001);
        assert_read_v([0b10101010,0b01010101], 3, 0b01010010);
        assert_read_v([0b10101010,0b01010101], 4, 0b10100101);
        assert_read_v([0b10101010,0b01010101], 5, 0b01001010);
        assert_read_v([0b10101010,0b01010101], 6, 0b10010101);
        assert_read_v([0b10101010,0b01010101], 7, 0b00101010);
        assert_read_v([0b10101010,0b01010101], 8, 0b01010101);

        assert_read_arr([0b10101010,0b01010101], 0, 0b10101010);
        assert_read_arr([0b10101010,0b01010101], 1, 0b01010100);
        assert_read_arr([0b10101010,0b01010101], 2, 0b10101001);
        assert_read_arr([0b10101010,0b01010101], 3, 0b01010010);
        assert_read_arr([0b10101010,0b01010101], 4, 0b10100101);
        assert_read_arr([0b10101010,0b01010101], 5, 0b01001010);
        assert_read_arr([0b10101010,0b01010101], 6, 0b10010101);
        assert_read_arr([0b10101010,0b01010101], 7, 0b00101010);
        assert_read_arr([0b10101010,0b01010101], 8, 0b01010101);
    }
    #[test]
    fn test_read_u16(){
        fn assert_read_v(data: impl AsRef<[u8]>,off: u8,expect: u16){
            assert_read(data,off,expect,|mut r|r.read_u16().unwrap());
        }
        fn assert_read_arr(data: impl AsRef<[u8]>,off: u8,expect: u16){
            assert_read(data,off,expect,|mut r|{
                let mut val = [0u8;2];
                r.read_exact(&mut val).map(|_|u16::from_be_bytes(val)).unwrap()
            });
        }
        assert_read_v([0b10101010,0b01010101,0b10101010], 0, 0b10101010_01010101);
        assert_read_v([0b10101010,0b01010101,0b10101010], 1, 0b01010100_10101011);
        assert_read_v([0b10101010,0b01010101,0b10101010], 2, 0b10101001_01010110);
        assert_read_v([0b10101010,0b01010101,0b10101010], 3, 0b01010010_10101101);
        assert_read_v([0b10101010,0b01010101,0b10101010], 4, 0b10100101_01011010);
        assert_read_v([0b10101010,0b01010101,0b10101010], 5, 0b01001010_10110101);
        assert_read_v([0b10101010,0b01010101,0b10101010], 6, 0b10010101_01101010);
        assert_read_v([0b10101010,0b01010101,0b10101010], 7, 0b00101010_11010101);
        assert_read_v([0b10101010,0b01010101,0b10101010], 8, 0b01010101_10101010);

        assert_read_arr([0b10101010,0b01010101,0b10101010], 0, 0b10101010_01010101);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 1, 0b01010100_10101011);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 2, 0b10101001_01010110);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 3, 0b01010010_10101101);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 4, 0b10100101_01011010);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 5, 0b01001010_10110101);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 6, 0b10010101_01101010);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 7, 0b00101010_11010101);
        assert_read_arr([0b10101010,0b01010101,0b10101010], 8, 0b01010101_10101010);
    }

    #[test]
    fn test_read_u32(){
        fn assert_read_v(data: impl AsRef<[u8]>,off: u8,expect: u32){
            assert_read(data,off,expect,|mut r|r.read_u32().unwrap());
        }
        fn assert_read_arr(data: impl AsRef<[u8]>,off: u8,expect: u32){
            assert_read(data,off,expect,|mut r|{
                let mut val = [0u8;4];
                r.read_exact(&mut val).map(|_|u32::from_be_bytes(val)).unwrap()
            });
        }
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 0, 0b10101010_01010101_10101010_01010101);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 1, 0b01010100_10101011_01010100_10101011);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 2, 0b10101001_01010110_10101001_01010110);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 3, 0b01010010_10101101_01010010_10101101);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 4, 0b10100101_01011010_10100101_01011010);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 5, 0b01001010_10110101_01001010_10110101);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 6, 0b10010101_01101010_10010101_01101010);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 7, 0b00101010_11010101_00101010_11010101);
        assert_read_v([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 8, 0b01010101_10101010_01010101_10101010);

        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 0, 0b10101010_01010101_10101010_01010101);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 1, 0b01010100_10101011_01010100_10101011);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 2, 0b10101001_01010110_10101001_01010110);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 3, 0b01010010_10101101_01010010_10101101);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 4, 0b10100101_01011010_10100101_01011010);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 5, 0b01001010_10110101_01001010_10110101);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 6, 0b10010101_01101010_10010101_01101010);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 7, 0b00101010_11010101_00101010_11010101);
        assert_read_arr([0b10101010,0b01010101,0b10101010,0b01010101,0b10101010], 8, 0b01010101_10101010_01010101_10101010);
    }
    #[test]
    fn test_read_exact(){
        let mut rand = &mut StdRng::seed_from_u64(1234);


        let mut data = Vec::with_capacity(1000);
        let mut exp = Vec::with_capacity(data.capacity());
        let mut buf = Vec::with_capacity(data.capacity());

        for _ in 0..3000 {
            data.clear();
            exp.clear();
            buf.clear();
            let count = rand.gen_range(2..data.capacity());
            data.extend(repeat_with(||rand.gen::<u8>()).take(count));
            exp.extend_from_slice(&data);
            let offset = rand.gen_range(0..8);


            for i in 0..(exp.len()-1) {
                let value = u16::from_be_bytes([exp[i],exp[i+1]]).wrapping_shl(offset);
                exp[i] = value.to_be_bytes()[0];
            }
            exp.pop();

            buf.extend(std::iter::repeat(0u8).take(exp.len()));
            assert_read(&data,offset as u8,&exp,|mut r|{
                r.read_exact(&mut buf).unwrap();
                &buf
            });
        }

    }

}