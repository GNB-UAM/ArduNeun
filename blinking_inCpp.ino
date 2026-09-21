void setup()
{
  pinMode( LED_BUILTIN, OUTPUT );
}
 
class Tblink
{
   public: 
   Tblink()  // constroctor
   {
          digitalWrite( LED_BUILTIN, LOW ); 
   }
 
   void on()
   {
        digitalWrite( LED_BUILTIN, HIGH );  
        delay(333); 
   }
 
   void off()
   {
        digitalWrite( LED_BUILTIN, LOW );  
        delay(333);   
   }
 
};
 
void loop()
{
      Tblink blink; // create a blink object with Tblink class Type, this will run its constructor method
 
      blink.on();  
      blink.off();           
}