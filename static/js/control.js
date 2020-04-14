$(document).ready(function() {
    $('#dataframe').DataTable( {
        "scrollY": true,
        "scrollX": true,
        "lengthMenu": [[10, 20, 50, -1], [10, 20, 50, "All"]]
    } );
} );
