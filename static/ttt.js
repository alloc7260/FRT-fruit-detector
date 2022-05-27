function loadFile(event){
	var inimage = document.getElementById('input');
	inimage.src = URL.createObjectURL(event.target.files[0]);
	var s = document.getElementById("js1");
	s.value = inimage.src;
	console.log(s.value);
};