$(document).ready(function () {

    //file upload example
    var container = $('#indicatorContainerWrap'),
        msgHolder = container.find('.rad-cntnt'),
        containerProg = container.radialIndicator({
            radius: 100,
            percentage: true,
            displayNumber: false
        }).data('radialIndicator');


    container.on({
        'dragenter': function (e) {
            msgHolder.html("Drop here");
        },
        'dragleave': function (e) {
            msgHolder.html("Click / Drop file to select.");
        },
        'drop': function (e) {
            e.preventDefault();
            handleFileUpload(e.originalEvent.dataTransfer.files);
        }
    });

    $('#prgFileSelector').on('change', function () {
        handleFileUpload(this.files);
    });

    function handleFileUpload(files) {
        msgHolder.hide();
        containerProg.option('displayNumber', true);

        var file = files[0],
            fd = new FormData();

        fd.append('file', file);


        $.ajax({
            url: 'service/upload.php',
            type: 'POST',
            data: fd,
            processData: false,
            contentType: false,
            success: function () {
                containerProg.option('displayNumber', false);
                msgHolder.show().html('File upload done.');
            },
            xhr: function () {
                var xhr = new window.XMLHttpRequest();
                //Upload progress
                xhr.upload.addEventListener("progress", function (e) {
                    if (e.lengthComputable) {
                        var percentComplete = (e.loaded || e.position) * 100 / e.total;
                        //Do something with upload progress
                        console.log(percentComplete);
                        containerProg.animate(percentComplete);
                    }
                }, false);

                return xhr;
            }
        });

    }


});

var loadFile = function(event) {
    var image = document.getElementById('output');
    image.src=URL.createObjectURL(event.target.files[0]);
};